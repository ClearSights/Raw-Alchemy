# -*- coding: utf-8 -*-
"""
Adobe XMP Profile Generator (Fixed to match XMPconverter.cpp)

Features:
1. Color Space Transform (Linear ProPhoto -> Target Log -> User LUT).
2. Tetrahedral Interpolation for high-quality resizing.
3. Adobe RGBTable Binary Format (Delta Encoded, Zlib Compressed, Base85).
4. Full range amount slider (0-200%).

Dependencies: pip install colour-science numpy
"""

import hashlib
import struct
import uuid
import zlib
from io import BytesIO

import numpy as np
import colour

# Handle optional dependencies for standalone usage
try:
    from .constants import LOG_ENCODING_MAP, LOG_TO_WORKING_SPACE
except ImportError:
    # Dummy maps if running standalone for testing
    LOG_ENCODING_MAP = {}
    LOG_TO_WORKING_SPACE = {}

# --- Constants & Mappings ---

# Adobe Custom Base85 Characters (Standard Adobe Order)
# C++ Source: kEncodeTable
ADOBE_Z85_CHARS = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?`'|()[]{}@%$#"
ADOBE_Z85_TABLE = [chr(c) for c in ADOBE_Z85_CHARS]

# --- Core Logic ---

def adobe_base85_encode(data: bytes) -> str:
    """
    Encodes binary data into Adobe's custom Base85 format.
    
    CRITICAL CHANGE: 
    Matches XMPconverter.cpp logic which uses Little-Endian reading 
    and LSB-first character generation (val % 85).
    """
    length = len(data)
    encoded_chars = []
    
    # Process 4-byte chunks
    for i in range(0, length, 4):
        chunk = data[i : i + 4]
        chunk_len = len(chunk)
        
        # Pad with null bytes if less than 4
        if chunk_len < 4:
            chunk = chunk + b'\x00' * (4 - chunk_len)
        
        # Unpack as Little-Endian Unsigned Int (<I)
        # C++ does: x = *(sPtr_1_ + i); which is LE on Intel/Windows
        val = struct.unpack('<I', chunk)[0]
        
        # Calculate 5 Base85 characters
        # C++ logic: for (j=0; j<5; ++j, x /= 85) dPtr_2[k++] = kEncodeTable[x % 85];
        # This outputs the LSB (remainder) first.
        for _ in range(5):
            encoded_chars.append(ADOBE_Z85_TABLE[val % 85])
            val //= 85

    # Handle padding length correction
    # If original length is not div by 4, we must output only the needed characters.
    # 1 byte  -> 2 chars
    # 2 bytes -> 3 chars
    # 3 bytes -> 4 chars
    if length % 4 != 0:
        rem = length % 4
        # Calculate how many chars to keep from the last block of 5
        chars_to_keep = rem + 1
        # Remove the extra characters from the end
        encoded_chars = encoded_chars[: -(5 - chars_to_keep)]

    return "".join(encoded_chars)

def int_round(arr):
    """Matches C++ int_round: floor(n + 0.5)"""
    return np.floor(arr + 0.5).astype(np.int32)

def apply_cst_pipeline(user_lut_path, log_space, output_size=33, _log=print):
    """
    Loads user LUT, creates ProPhoto Identity, transforms to Log, applies LUT.
    Returns: (output_size, final_data_numpy)
    """
    _log(f"Processing: Reading {user_lut_path}...")
    user_lut = colour.read_LUT(user_lut_path)
    
    # --- FIX START ---
    # 1. Create Identity Grid in Linear ProPhoto RGB
    # 修正：使用 (R, G, B) 顺序。Axis 0 是 R，Axis 2 是 B。
    # 这符合 colour-science 和 .cube 文件的标准顺序。
    domain = np.linspace(0, 1, output_size)
    R, G, B = np.meshgrid(domain, domain, domain, indexing='ij')
    
    # 堆叠为 (R, G, B, 3) 形状
    prophoto_linear = np.stack([R, G, B], axis=-1) 
    # --- FIX END ---
    
    log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
    log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)

    _log(f"  - Pipeline: ProPhoto Linear -> {log_color_space_name} -> {log_curve_name} -> LUT")
        
    # A. Gamut Transform: ProPhoto RGB -> Target Gamut (Linear)
    matrix = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[log_color_space_name]
    )
    # 矩阵乘法 (Numpy array 是行向量，所以乘转置矩阵)
    target_gamut_linear = prophoto_linear @ matrix.T
    target_gamut_linear = np.maximum(target_gamut_linear, 1e-7)

    # B. Transfer Function: Linear -> Log
    log_encoded = colour.cctf_encoding(target_gamut_linear, function=log_curve_name)
        
    # C. Apply User LUT
    # 由于现在的 grid 结构是 (R,G,B)，与 standard LUT 结构一致，插值结果也会保持正确的空间顺序
    _log(f"  - Applying User LUT ({user_lut.size}^3) to grid...")
    final_rgb = user_lut.apply(log_encoded, interpolator=colour.algebra.table_interpolation_tetrahedral)
    
    # --- Debug Feature ---
    try:
        debug_filename = f"debug_pipeline_{output_size}.cube"
        _log(f"  [DEBUG] Writing pipeline output to {debug_filename}...")
        # 此时 final_rgb 已经是标准的 (R, G, B, 3) 顺序，可以直接写入
        debug_lut = colour.LUT3D(table=final_rgb, name=f"Debug Pipeline {log_space}")
        colour.write_LUT(debug_lut, debug_filename)
        _log(f"  [DEBUG] Successfully wrote {debug_filename}")
    except Exception as e:
        _log(f"  [DEBUG] Failed to write debug cube: {e}")

    return output_size, final_rgb

def generate_rgb_table_stream(data, size, min_amt=0, max_amt=200):
    """
    Encodes the numpy data into DNG RGBTable binary format.
    Input data shape MUST be: (R, G, B, 3)
    """
    stream = BytesIO()
    def write_u32(val): stream.write(struct.pack('<I', val))
    def write_double(val): stream.write(struct.pack('<d', val))
    
    # Header
    write_u32(1) # btt_RGBTable
    write_u32(1) # Version
    write_u32(3) # Dimensions
    write_u32(size) # Divisions
    
    # 1. Clip and Scale
    data = np.clip(data, 0.0, 1.0)
    data_u16 = int_round(data * 65535) # Shape (R, G, B, 3)
    
    # 2. Prepare Identity Curve
    indices = np.arange(size, dtype=np.int32)
    # Standard DNG Identity logic
    nop_curve = (indices * 0xFFFF + (size >> 1)) // (size - 1)
    
    # 3. Prepare Nop Grid matching Data Shape (R, G, B)
    # indexing='ij' -> Axis 0=R, 1=G, 2=B.
    grid_r, grid_g, grid_b = np.meshgrid(nop_curve, nop_curve, nop_curve, indexing='ij')
    
    # 4. Calculate Deltas (Value - Identity)
    delta_r = data_u16[..., 0] - grid_r
    delta_g = data_u16[..., 1] - grid_g
    delta_b = data_u16[..., 2] - grid_b
    
    # 5. Interleave and Flatten
    # C++ Loop Order: bIndex(outer), gIndex, rIndex(inner).
    # Memory Layout: This implies Row-Major (C-Style) flattening of an (R, G, B) block matches.
    # Stack along last axis -> (R, G, B, 3)
    deltas_stacked = np.stack((delta_r, delta_g, delta_b), axis=-1)
    
    # Flatten to 1D array
    # Since we are casting to uint16, we need to handle negative values (deltas)
    # by simulating overflow/cast. 
    flat_deltas = deltas_stacked.flatten().astype(np.int32)
    
    # Convert to 16-bit bytes (Little Endian)
    # astype('<u2') will interpret the lower 16 bits of the int32, correctly handling negative 2's complement
    stream.write(flat_deltas.astype('<u2').tobytes())
        
    # Footer
    write_u32(0) # sRGB
    write_u32(1) # sRGB (Gamma)
    write_u32(0) # Gamut Extend (0=Clip)
    write_double(min_amt * 0.01)
    write_double(max_amt * 0.01)
    
    return stream.getvalue()

def create_xmp_profile(profile_name, lut_path, log_space=None, _log=print):
    profile_uuid = str(uuid.uuid4()).replace('-', '').upper()
    
    try:
        # Standard size for DNG RGBTable is often 32 or 33.
        size, data = apply_cst_pipeline(lut_path, log_space, output_size=33, _log=_log)
        
        # Binary Encoding
        raw_bytes = generate_rgb_table_stream(data, size, min_amt=0, max_amt=200)
        
        # Fingerprinting (MD5 of uncompressed binary)
        m = hashlib.md5()
        m.update(raw_bytes)
        fingerprint = m.hexdigest().upper()
        
        # Compression & ASCII Encoding
        # 1. Prefix with original length (4 bytes Little Endian)
        # Matches C++: memcpy(dPtr_1, &uncompressedSize_1, 4) -> LE on x86
        header = struct.pack('<I', len(raw_bytes))
        
        # 2. Compress payload
        compressed = zlib.compress(raw_bytes, level=zlib.Z_BEST_COMPRESSION)
        
        # 3. Base85 Encode the whole thing (Header + Compressed Data)
        encoded_data = adobe_base85_encode(header + compressed)
        
    except Exception as e:
        _log(f"Error creating profile: {e}")
        import traceback
        traceback.print_exc()
        return ""

    xmp_template = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:PresetType="Look"
   crs:Cluster=""
   crs:UUID="{profile_uuid}"
   crs:SupportsAmount="True"
   crs:SupportsColor="True"
   crs:SupportsMonochrome="True"
   crs:SupportsHighDynamicRange="True"
   crs:SupportsNormalDynamicRange="True"
   crs:SupportsSceneReferred="True"
   crs:SupportsOutputReferred="True"
   crs:RequiresRGBTables="False"
   crs:ShowInPresets="True"
   crs:ShowInQuickActions="False"
   crs:CameraModelRestriction=""
   crs:Copyright=""
   crs:ContactInfo=""
   crs:Version="14.3"
   crs:ProcessVersion="11.0"
   crs:ConvertToGrayscale="False"
   crs:RGBTable="{fingerprint}"
   crs:Table_{fingerprint}="{encoded_data}"
   crs:HasSettings="True">
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{profile_name}</rdf:li>
    </rdf:Alt>
   </crs:Name>
   <crs:Group>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">Profiles</rdf:li>
    </rdf:Alt>
   </crs:Group>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    return xmp_template