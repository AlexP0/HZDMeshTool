bl_info = {
    "name": "HZD Mesh Tool",
    "author": "AlexPo",
    "location": "Scene Properties > HZD Panel",
    "version": (1, 4, 0),
    "blender": (3, 2, 0),
    "description": "This addon imports/exports skeletal meshes\n from Horizon Zero Dawn's .core/.stream files",
    "category": "Import-Export"
    }


if "bpy" in locals():
    import imp
    imp.reload(pymmh3)
else:
    from . import pymmh3

import bpy
import bmesh
import os
import pathlib
from struct import unpack, pack
import numpy as np
import mathutils
import math
import operator

from sys import platform
import ctypes
from ctypes import c_size_t, c_char_p, c_int32
from pathlib import Path
from typing import Union, Dict
from enum import IntEnum
import subprocess
# python debuger
import pdb

class DXGI(IntEnum):
    DXGI_FORMAT_UNKNOWN = 0,
    DXGI_FORMAT_R32G32B32A32_TYPELESS = 1,
    DXGI_FORMAT_R32G32B32A32_FLOAT = 2,
    DXGI_FORMAT_R32G32B32A32_UINT = 3,
    DXGI_FORMAT_R32G32B32A32_SINT = 4,
    DXGI_FORMAT_R32G32B32_TYPELESS = 5,
    DXGI_FORMAT_R32G32B32_FLOAT = 6,
    DXGI_FORMAT_R32G32B32_UINT = 7,
    DXGI_FORMAT_R32G32B32_SINT = 8,
    DXGI_FORMAT_R16G16B16A16_TYPELESS = 9,
    DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
    DXGI_FORMAT_R16G16B16A16_UNORM = 11,
    DXGI_FORMAT_R16G16B16A16_UINT = 12,
    DXGI_FORMAT_R16G16B16A16_SNORM = 13,
    DXGI_FORMAT_R16G16B16A16_SINT = 14,
    DXGI_FORMAT_R32G32_TYPELESS = 15,
    DXGI_FORMAT_R32G32_FLOAT = 16,
    DXGI_FORMAT_R32G32_UINT = 17,
    DXGI_FORMAT_R32G32_SINT = 18,
    DXGI_FORMAT_R32G8X24_TYPELESS = 19,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT = 20,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS = 21,
    DXGI_FORMAT_X32_TYPELESS_G8X24_UINT = 22,
    DXGI_FORMAT_R10G10B10A2_TYPELESS = 23,
    DXGI_FORMAT_R10G10B10A2_UNORM = 24,
    DXGI_FORMAT_R10G10B10A2_UINT = 25,
    DXGI_FORMAT_R11G11B10_FLOAT = 26,
    DXGI_FORMAT_R8G8B8A8_TYPELESS = 27,
    DXGI_FORMAT_R8G8B8A8_UNORM = 28,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29,
    DXGI_FORMAT_R8G8B8A8_UINT = 30,
    DXGI_FORMAT_R8G8B8A8_SNORM = 31,
    DXGI_FORMAT_R8G8B8A8_SINT = 32,
    DXGI_FORMAT_R16G16_TYPELESS = 33,
    DXGI_FORMAT_R16G16_FLOAT = 34,
    DXGI_FORMAT_R16G16_UNORM = 35,
    DXGI_FORMAT_R16G16_UINT = 36,
    DXGI_FORMAT_R16G16_SNORM = 37,
    DXGI_FORMAT_R16G16_SINT = 38,
    DXGI_FORMAT_R32_TYPELESS = 39,
    DXGI_FORMAT_D32_FLOAT = 40,
    DXGI_FORMAT_R32_FLOAT = 41,
    DXGI_FORMAT_R32_UINT = 42,
    DXGI_FORMAT_R32_SINT = 43,
    DXGI_FORMAT_R24G8_TYPELESS = 44,
    DXGI_FORMAT_D24_UNORM_S8_UINT = 45,
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS = 46,
    DXGI_FORMAT_X24_TYPELESS_G8_UINT = 47,
    DXGI_FORMAT_R8G8_TYPELESS = 48,
    DXGI_FORMAT_R8G8_UNORM = 49,
    DXGI_FORMAT_R8G8_UINT = 50,
    DXGI_FORMAT_R8G8_SNORM = 51,
    DXGI_FORMAT_R8G8_SINT = 52,
    DXGI_FORMAT_R16_TYPELESS = 53,
    DXGI_FORMAT_R16_FLOAT = 54,
    DXGI_FORMAT_D16_UNORM = 55,
    DXGI_FORMAT_R16_UNORM = 56,
    DXGI_FORMAT_R16_UINT = 57,
    DXGI_FORMAT_R16_SNORM = 58,
    DXGI_FORMAT_R16_SINT = 59,
    DXGI_FORMAT_R8_TYPELESS = 60,
    DXGI_FORMAT_R8_UNORM = 61,
    DXGI_FORMAT_R8_UINT = 62,
    DXGI_FORMAT_R8_SNORM = 63,
    DXGI_FORMAT_R8_SINT = 64,
    DXGI_FORMAT_A8_UNORM = 65,
    DXGI_FORMAT_R1_UNORM = 66,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP = 67,
    DXGI_FORMAT_R8G8_B8G8_UNORM = 68,
    DXGI_FORMAT_G8R8_G8B8_UNORM = 69,
    DXGI_FORMAT_BC1_TYPELESS = 70,
    DXGI_FORMAT_BC1_UNORM = 71,
    DXGI_FORMAT_BC1_UNORM_SRGB = 72,
    DXGI_FORMAT_BC2_TYPELESS = 73,
    DXGI_FORMAT_BC2_UNORM = 74,
    DXGI_FORMAT_BC2_UNORM_SRGB = 75,
    DXGI_FORMAT_BC3_TYPELESS = 76,
    DXGI_FORMAT_BC3_UNORM = 77,
    DXGI_FORMAT_BC3_UNORM_SRGB = 78,
    DXGI_FORMAT_BC4_TYPELESS = 79,
    DXGI_FORMAT_BC4_UNORM = 80,
    DXGI_FORMAT_BC4_SNORM = 81,
    DXGI_FORMAT_BC5_TYPELESS = 82,
    DXGI_FORMAT_BC5_UNORM = 83,
    DXGI_FORMAT_BC5_SNORM = 84,
    DXGI_FORMAT_B5G6R5_UNORM = 85,
    DXGI_FORMAT_B5G5R5A1_UNORM = 86,
    DXGI_FORMAT_B8G8R8A8_UNORM = 87,
    DXGI_FORMAT_B8G8R8X8_UNORM = 88,
    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM = 89,
    DXGI_FORMAT_B8G8R8A8_TYPELESS = 90,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91,
    DXGI_FORMAT_B8G8R8X8_TYPELESS = 92,
    DXGI_FORMAT_B8G8R8X8_UNORM_SRGB = 93,
    DXGI_FORMAT_BC6H_TYPELESS = 94,
    DXGI_FORMAT_BC6H_UF16 = 95,
    DXGI_FORMAT_BC6H_SF16 = 96,
    DXGI_FORMAT_BC7_TYPELESS = 97,
    DXGI_FORMAT_BC7_UNORM = 98,
    DXGI_FORMAT_BC7_UNORM_SRGB = 99,
    DXGI_FORMAT_AYUV = 100,
    DXGI_FORMAT_Y410 = 101,
    DXGI_FORMAT_Y416 = 102,
    DXGI_FORMAT_NV12 = 103,
    DXGI_FORMAT_P010 = 104,
    DXGI_FORMAT_P016 = 105,
    DXGI_FORMAT_420_OPAQUE = 106,
    DXGI_FORMAT_YUY2 = 107,
    DXGI_FORMAT_Y210 = 108,
    DXGI_FORMAT_Y216 = 109,
    DXGI_FORMAT_NV11 = 110,
    DXGI_FORMAT_AI44 = 111,
    DXGI_FORMAT_IA44 = 112,
    DXGI_FORMAT_P8 = 113,
    DXGI_FORMAT_A8P8 = 114,
    DXGI_FORMAT_B4G4R4A4_UNORM = 115,
    DXGI_FORMAT_P208 = 130,
    DXGI_FORMAT_V208 = 131,
    DXGI_FORMAT_V408 = 132,
    DXGI_FORMAT_FORCE_UINT = 0xffffffff

BoneMatrices = {} #TODO this BoneMatrices is very weird.

verbose = True

def say(string,level=0):
    if verbose:
        print(str(string))

# Path to "NVIDIA Texture Tools Exporter" for converting DDS to PNG
if any([platform.startswith(os_name) for os_name in ['linux', 'darwin', 'freebsd']]):
    NVTTDefaultPath = Path('/opt/NVIDIA_Texture_Tools_Linux_3_1_6/nvdecompress')
else:
    NVTTDefaultPath = Path('C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools/nvtt_export.exe')

# -----------------------------------------------------------------------------
# Node Adding Operator
#
class NODE_OT_HZDMT_add(bpy.types.Operator):
    """Add a HZD Mesh Tool node group"""
    bl_idname = "node.hzdmt_add"
    bl_label = "Add HZD Mesh Tool node group"
    bl_description = "Add HZD Mesh Tool node group"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        subtype='FILE_PATH',
    )
    group_name: bpy.props.StringProperty()

    def execute(self, context):
        if bpy.data.node_groups.find(self.group_name) == -1:
            with bpy.data.libraries.load(self.filepath, link=True) as (data_from, data_to):
                assert(self.group_name in data_from.node_groups)
                data_to.node_groups = [self.group_name]
        node_type = {
            "ShaderNodeTree": "ShaderNodeGroup",
            "CompositorNodeTree": "CompositorNodeGroup",
            "TextureNodeTree": "TextureNodeGroup",
            "GeometryNodeTree": "GeometryNodeGroup",
        }[type(context.space_data.edit_tree).__name__]
        bpy.ops.node.add_node('INVOKE_DEFAULT', type=node_type, use_transform=True, settings=[{"name":"node_tree", "value":"bpy.data.node_groups['" + self.group_name + "']"}])
        return {'FINISHED'}

#
# Node menu list
#
def node_hzdmt_cache(reload=False):
    dirpath = os.path.dirname(__file__) # addon path

    if node_hzdmt_cache._node_cache_path != dirpath:
        reload = True

    node_cache = node_hzdmt_cache._node_cache
    if reload:
        node_cache = []
    if node_cache:
        return node_cache

    for filepath in Path(dirpath).rglob('*.blend'):
        with bpy.data.libraries.load(str(filepath)) as (data_from, data_to):
            for group_name in data_from.node_groups:
                node_cache.append((group_name, str(filepath)))

    node_cache = sorted(node_cache)

    node_hzdmt_cache._node_cache = node_cache
    node_hzdmt_cache._node_cache_path = dirpath

    return node_cache

node_hzdmt_cache._node_cache = []
node_hzdmt_cache._node_cache_path = ""

class NODE_MT_HZDMT_add(bpy.types.Menu):
    bl_label = "Node HZD Mesh Tool"

    def draw(self, context):
        layout = self.layout

        try:
            node_items = node_hzdmt_cache()
        except Exception as ex:
            node_items = ()
            layout.label(text=repr(ex), icon='ERROR')

        for group_name, filepath in node_items:
            if not group_name.startswith("_"):
                props = layout.operator(
                    NODE_OT_HZDMT_add.bl_idname,
                    text=group_name,
                )
                props.filepath = filepath
                props.group_name = group_name

def add_node_button(self, context):
    self.layout.menu(
        NODE_MT_HZDMT_add.__name__,
        text="HZD Mesh Tool",
        icon='PLUGIN',
    )
# -----------------------------------------------------------------------------

class ArchiveManager:
    class BinHeader:
        def __init__(self):
            self.version = 0
            self.key = 0
            self.filesize = 0
            self.datasize = 0
            self.filecount = 0
            self.chunkcount = 0
            self.maxchunksize = 0

        def parse(self, f):
            r = ByteReader
            self.version = r.int32(f)
            self.key = r.int32(f)
            self.filesize = r.uint64(f)
            self.datasize = r.uint64(f)
            self.filecount = r.uint64(f)
            self.chunkcount = r.int32(f)
            self.maxchunksize = r.int32(f)

        def print(self):
            print("Header", "\n",
                  "Ver = ", self.version, "\n",
                  "Key = ", self.key, "\n",
                  "FileSize = ", self.filesize, "\n",
                  "DataSize = ", self.datasize, "\n",
                  "FileCount = ", self.filecount, "\n",
                  "ChunkCount = ", self.chunkcount, "\n",
                  "MaxChunkSize =", self.maxchunksize)
    class FileEntry:
        def __init__(self):
            self.id = 0
            self.key0 = 0
            self.hash = 0
            self.offset = 0
            self.size = 0
            self.key1 = 0

        def parse(self, f):
            r = ByteReader
            self.id = r.uint32(f)
            self.key0 = r.uint32(f)
            self.hash = r.uint64(f)
            self.offset = r.uint64(f)
            self.size = r.uint32(f)
            self.key1 = r.uint32(f)

        def print(self):
            print("File", "\n",
                  "ID = ", self.id, "\n",
                  "Key0 = ", self.key0, "\n",
                  "Hash = ", self.hash, "\n",
                  "Offset = ", self.offset, "\n",
                  "Size = ", self.size, "\n",
                  "Key1 = ", self.key1, "\n")
    class ChunkEntry:
        def __init__(self):
            self.uncompressed_offset = 0
            self.uncompressed_size = 0
            self.key0 = 0
            self.compressed_offset = 0
            self.compressed_size = 0
            self.key1 = 0

        def parse(self, f):
            r = ByteReader
            self.uncompressed_offset = r.uint64(f)
            self.uncompressed_size = r.uint32(f)
            self.key0 = r.int32(f)
            self.compressed_offset = r.uint64(f)
            self.compressed_size = r.uint32(f)
            self.key1 = r.uint32(f)

        def write(self, f):
            w = BytePacker
            bChunkEntry = b''
            bChunkEntry += w.uint64(self.uncompressed_offset)
            bChunkEntry += w.int32(self.uncompressed_size)
            bChunkEntry += w.int32(self.key0)
            bChunkEntry += w.uint64(self.compressed_offset)
            bChunkEntry += w.int32(self.compressed_size)
            bChunkEntry += w.int32(self.key1)
            f.write(bChunkEntry)

        def print(self):
            print("Chunk", "\n",
                  "U Offset = ", self.uncompressed_offset, "\n",
                  "U Size = ", self.uncompressed_size, "\n",
                  "Key0 = ", self.key0, "\n",
                  "C Offset = ", self.compressed_offset, "\n",
                  "C Size = ", self.compressed_size, "\n",
                  "Key1 = ", self.key1, "\n")

    def __init__(self):
        self.DataStart = 0
        self.Chunks = []
        self.DesiredArchive = ""


    @staticmethod
    def get_file_hash(string):
        tmp = string.encode("utf8") + b'\x00'
        fileHash = pymmh3.hash64(tmp, 42, True)[0]
        # print(hex(fileHash), fileHash)
        # bHash = BytePacker.uint64(fileHash)
        # print(bHash)
        say("mmh3 = "+hex(fileHash)+" ("+string+")")
        return fileHash

    def FindChunkContainingOffset(self,Uoffset):
        # say("looking for chunk containing offset:"+str(Uoffset))
        for i, c in enumerate(self.Chunks):
            if Uoffset in range(c.uncompressed_offset, c.uncompressed_offset + c.uncompressed_size):
                # print(c.uncompressed_offset, c.uncompressed_offset + c.uncompressed_size,i)
                return i


    def ClipChunk(self,file, StartChunkIndex):
        RealStartOffset = file.offset - self.Chunks[StartChunkIndex].uncompressed_offset
        RealEndOffset = RealStartOffset + file.size
        # print(file.offset,Chunks[StartChunkIndex].uncompressed_offset,RealStartOffset,RealEndOffset)
        return RealStartOffset, RealEndOffset
    def GetExtractedFilePath(self,filePath,isStream=False):
        HZDEditor = bpy.context.scene.HZDEditor
        if filePath[-5:] == ".core" or filePath[-12:] == ".core.stream":
            ExtractedFilePath = HZDEditor.WorkAbsPath + filePath
        else:
            if isStream:
                ExtractedFilePath = HZDEditor.WorkAbsPath + filePath + ".core.stream"
            else:
                ExtractedFilePath = HZDEditor.WorkAbsPath + filePath + ".core"
        return ExtractedFilePath
    def ExtractFile(self,file,filePath, isStream = False):
        class Oodle:
            HZDEditor = bpy.context.scene.HZDEditor

            # _local_path = Path(__file__).absolute().parent
            if any([platform.startswith(os_name) for os_name in ['linux', 'darwin', 'freebsd']]):
                _lib = ctypes.CDLL(str(HZDEditor.GameAbsPath + "/" + 'liboo2corelinux64.so.9'))
            else:
                _lib = ctypes.WinDLL(str(HZDEditor.GameAbsPath + "/" + 'oo2core_3_win64.dll'))
            # _lib = ctypes.WinDLL("S:\SteamLibrary\steamapps\common\Horizon Zero Dawn\oo2core_3_win64.dll")
            _compress = _lib.OodleLZ_Compress
            _compress.argtypes = [c_int32, c_char_p, c_size_t, c_char_p, c_int32, c_size_t, c_size_t, c_size_t,
                                  c_size_t,
                                  c_size_t]
            _compress.restype = c_int32
            _decompress = _lib.OodleLZ_Decompress
            _decompress.argtypes = [c_char_p, c_size_t, c_char_p, c_size_t, c_int32, c_int32, c_int32, c_size_t,
                                    c_size_t,
                                    c_size_t, c_size_t, c_size_t, c_size_t, c_int32]
            _decompress.restype = c_int32

            @staticmethod
            def decompress(input_buffer: Union[bytes, bytearray], output_size):
                out_data_p = ctypes.create_string_buffer(output_size)
                in_data_p = ctypes.create_string_buffer(bytes(input_buffer))
                result = Oodle._decompress(in_data_p, len(input_buffer), out_data_p, output_size, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0)
                assert result >= 0, 'Error decompressing chunk'
                return bytes(out_data_p)

            @staticmethod
            def compress(input_buffer: Union[bytes, bytearray], fmt: int = 8, level: int = 4):
                def calculate_compression_bound(size):
                    return size + 274 * ((size + 0x3FFFF) // 0x40000)

                out_size = calculate_compression_bound(len(input_buffer))
                out_data_p = ctypes.create_string_buffer(out_size)
                in_data_p = ctypes.create_string_buffer(bytes(input_buffer))

                result = Oodle._compress(fmt, in_data_p, len(input_buffer), out_data_p, level, 0, 0, 0, 0, 0)
                assert result >= 0, 'Error compressing chunk'
                return bytes(out_data_p[:result])
        if os.path.exists(bpy.context.scene.HZDEditor.GameAbsPath):
            say("Game Path is Valid")
            ## I do not know who the original author of this Oodle class is
            ## I got it from here https://github.com/REDxEYE/ProjectDecima_python/tree/master/ProjectDecima/utils
        else:
            raise Exception("Game Path is invalid")
        oodle = Oodle()
        HZDEditor = bpy.context.scene.HZDEditor



        DataChunks = b''
        if filePath[-5:]==".core" or filePath[-12:] == ".core.stream":
            ExtractedFilePath = HZDEditor.WorkAbsPath+filePath
        else:
            if isStream:
                ExtractedFilePath = HZDEditor.WorkAbsPath+filePath+".core.stream"
            else:
                ExtractedFilePath = HZDEditor.WorkAbsPath + filePath + ".core"

        if os.path.exists(ExtractedFilePath):
            say(filePath + "------Asset already extracted")
            return ExtractedFilePath
        else:
            say(filePath + "------Extracting Asset")
            StartChunkIndex = self.FindChunkContainingOffset(file.offset)
            EndChunkIndex = self.FindChunkContainingOffset(file.offset + file.size)

            directory = pathlib.Path(ExtractedFilePath).parent
            pathlib.Path(directory).mkdir(parents= True,exist_ok=True)
            with open(HZDEditor.GamePath + "Packed_DX12/" + self.DesiredArchive, 'rb') as f, open(ExtractedFilePath, 'wb') as w:
                for chunk in self.Chunks[StartChunkIndex:EndChunkIndex + 1]:
                    # chunk.print()
                    f.seek(chunk.compressed_offset)
                    buffer = f.read(chunk.compressed_size)
                    data = oodle.decompress(buffer, chunk.uncompressed_size)
                    DataChunks += data
                Start, End = self.ClipChunk(file, StartChunkIndex)
                w.write(DataChunks[Start:End])
                say("Created File: "+ ExtractedFilePath)
            return ExtractedFilePath
    def FindFile(self,filePath):


        HZDEditor = bpy.context.scene.HZDEditor

        # DesiredHash = b'\x0A\x4C\xD6\x5C\xF6\x5A\xFF\x2F' #Prefetch
        DesiredHash = self.get_file_hash(filePath)
        say(str(DesiredHash))
        for binArchive in ['Patch.bin','Remainder.bin','DLC1.bin','Initial.bin']:
            say("Searching for "+filePath+" in "+binArchive)
            with open(HZDEditor.GamePath + "Packed_DX12/" + binArchive, 'rb') as f:
                H = self.BinHeader()
                self.Chunks.clear()
                H.parse(f)
                # H.print()

                self.DataStart = (32 * H.filecount) + (32 * H.chunkcount) + 40
                # say("Data Start Offset: "+str(self.DataStart))
                foundFile = False
                for files in range(H.filecount):
                    file = self.FileEntry()
                    file.parse(f)
                    # Files.append(file)
                    # print(file.hash,DesiredHash)
                    if file.hash == DesiredHash:
                        file.print()
                        foundFile = True
                        DesiredFile = file
                if foundFile:
                    for chunk in range(H.chunkcount):
                        chunk = self.ChunkEntry()
                        chunk.parse(f)
                        self.Chunks.append(chunk)
                        self.DesiredArchive = binArchive
                    break
                else:
                    pass
        if not foundFile:
            raise Exception("Could not find file in bin archive.",filePath,DesiredHash)
        else:
            return DesiredFile
    def isFileInWorkspace(self,filePath,isStream):
        return os.path.exists(self.GetExtractedFilePath(filePath,isStream))

    def FindAndExtract(self,filePath,isStream = False):
        if self.isFileInWorkspace(filePath,isStream):
            assetFile = self.FindFile(filePath)
            ExtractedFilePath = self.ExtractFile(assetFile,filePath,isStream)
        else:
            ExtractedFilePath = self.GetExtractedFilePath(filePath,isStream)
            print("File Already Extracted: ",ExtractedFilePath)
        return ExtractedFilePath
def ClearProperties(self,context):
    HZDEditor = bpy.context.scene.HZDEditor
    HZDEditor.HZDAbsPath = bpy.path.abspath(HZDEditor.HZDPath)
    HZDEditor.GameAbsPath = bpy.path.abspath(HZDEditor.GamePath)
    HZDEditor.WorkAbsPath = bpy.path.abspath(HZDEditor.WorkPath)
    HZDEditor.SkeletonAbsPath = bpy.path.abspath(HZDEditor.SkeletonPath)
    # HZDEditor.SkeletonName = "Unknown Skeleton: Import Skeleton to set."
    BoneMatrices.clear()
    return None
class ByteReader:
    @staticmethod
    def int8(f):
        b = f.read(1)
        i = unpack('<b', b)[0]
        return i
    @staticmethod
    def bool(f):
        b = f.read(1)
        i = unpack('<b', b)[0]
        if i == 0:
            return False
        elif i == 1:
            return True
        else:
            raise Exception("Byte at {v} wasn't a boolean".format(v=f.tell()))
    @staticmethod
    def uint8(f):
        b = f.read(1)
        i = unpack('<B', b)[0]
        return i
    @staticmethod
    def int16(f):
        return unpack('<h', f.read(2))[0]
    @staticmethod
    def uint16(f):
        b = f.read(2)
        i = unpack('<H', b)[0]
        return i
    @staticmethod
    def float16(f):
        b = f.read(2)
        # print(b.hex())
        return float(np.frombuffer(b,dtype=np.float16)[0])
    @staticmethod
    def hash(f):
        b = f.read(8)
        return b
    @staticmethod
    def guid(f):
        return f.read(16)
    @staticmethod
    def int32(f):
        b = f.read(4)
        i = unpack('<i',b)[0]
        return i
    @staticmethod
    def uint32(f):
        b = f.read(4)
        i = unpack('<I',b)[0]
        return i
    @staticmethod
    def uint64(f):
        b = f.read(8)
        i = unpack('<Q',b)[0]
        return i
    @staticmethod
    def int64(f):
        b = f.read(8)
        i = unpack('<q', b)[0]
        return i
    @staticmethod
    def string(f,length):
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def path(f):
        b = f.read(4)
        length = unpack('<i', b)[0]
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def hashtext(f):
        b = f.read(4)
        length = unpack('<i', b)[0]
        f.seek(4,1)
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def float(f):
        b = f.read(4)
        fl = unpack('<f',b)[0]
        return fl
    @staticmethod
    def vector3(f):
        b = f.read(12)
        return unpack('<fff', b)
    @staticmethod
    def dvector3(f):
        #double vector 3
        b = f.read(24)
        return unpack('<ddd', b)
    @staticmethod
    def vector4(f):
        b = f.read(16)
        return unpack('<ffff', b)
    @staticmethod
    def int16Norm(f):
        i = unpack('<H', f.read(2))[0]
        v = i ^ 2**15
        v -= 2**15
        v /= 2**15 - 1
        return v
    @staticmethod
    def uint16Norm(f):
        int16 = unpack('<H', f.read(2))[0]
        return int16 / 2 ** 16
    @staticmethod
    def uint8Norm(f):
        int16 = unpack('<B', f.read(1))[0]
        maxint = 2 ** 8
        return int16 / maxint
    @staticmethod
    def X10Y10Z10W2Normalized(f):
        i = unpack('<I', f.read(4))[0]  # get 32bits of data

        x = i >> 0
        x = ((x & 0x3FF) ^ 512) - 512

        y = i >> 10
        y = ((y & 0x3FF) ^ 512) - 512

        z = i >> 20
        z = ((z & 0x3FF) ^ 512) - 512

        w = i >> 30
        w = w & 0x1

        vectorLength = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        # # print(x,y,z)
        if vectorLength != 0:
            x /= vectorLength
            y /= vectorLength
            z /= vectorLength
        return [x, y, z, w]
    @staticmethod
    def readVertexStorageType(f,storageType):
        ST = StreamData.VertexElementDesc.StorageType
        r = ByteReader
        if storageType == ST.Undefined:
            raise Exception("Undefined Storage Type at offset %d"%f.tell())
        if storageType == ST.SignedShortNormalized:
            return r.int16Norm(f)
        if storageType == ST.Float:
            return r.float(f)
        if storageType == ST.HalfFloat:
            return r.float16(f)
        if storageType == ST.UnsignedByteNormalized:
            return r.uint8Norm(f)
        if storageType == ST.SignedShort:
            return r.int8(f)
        if storageType == ST.X10Y10Z10W2Normalized:
            return r.X10Y10Z10W2Normalized(f)
        if storageType == ST.UnsignedByte:
            return r.uint8(f)
        if storageType == ST.UnsignedShort:
            return r.uint16(f)
        if storageType == ST.UnsignedShortNormalized:
            return r.uint16Norm(f)
        if storageType == ST.UNorm8sRGB:
            raise Exception("Storage Type not handled at offset %d" % f.tell())
        if storageType == ST.X10Y10Z10W2UNorm:
            raise Exception("Storage Type not handled at offset %d" % f.tell())

class BytePacker:
    @staticmethod
    def int8(v):
        return pack('<b', v)
    @staticmethod
    def uint8(v):
        return pack('<B', v)
    @staticmethod
    def uint8Norm(v):
        if 0.0 <= v <= 1.0:
            i = int(v * (2 ** 8))
        else:
            raise Exception("Couldn't normalize value as uint16Norm, "
                            "it wasn't between -1.0 and 1.0. Unknown max value."
                            +str(v))
        return pack('<B', i)
    @staticmethod
    def int16(v):
        return pack('<h', v)
    @staticmethod
    def uint16(v):
        return pack('<H', v)
    @staticmethod
    def int16Norm(v):
        if -1.0 < v < 1.0:
            if v >= 0:
                v = int(abs(v) * (2 ** 15))
            else:
                v = 2 ** 16 - int(abs(v) * (2 ** 15))
        else:
            raise Exception("Couldn't normalize value as int16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<H', v)
    @staticmethod
    def uint16Norm(v):
        if 0.0 < v < 1.0:
            i = v * (2 ** 16) - 1
        else:
            raise Exception("Couldn't normalize value as uint16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<H', i)
    @staticmethod
    def float16(v):
        f32 = np.float32(v)
        f16 = f32.astype(np.float16)
        b16 = f16.tobytes()
        return b16
    @staticmethod
    def int32(v):
        return pack('<i', v)
    @staticmethod
    def uint32(v):
        return pack('<I', v)
    @staticmethod
    def uint64(v):
        return pack('<Q', v)
    @staticmethod
    def int64(v):
        return pack('<q', v)
    @staticmethod
    def float(v):
        return pack('<f', v)
    @staticmethod
    def X10Y10Z10W2(x,y,z,w):
        if x >= 0:
            x = int(abs(x) * 2 ** 9)
        else:
            x = 2**10 - int(abs(x) * 2 ** 9)
        if y >= 0:
            y = int(abs(y) * 2 ** 9)
        else:
            y = 2**10 - int(abs(y) * 2 ** 9)
        if z >= 0:
            z = int(abs(z) * 2 ** 9)
        else:
            z = 2**10 - int(abs(z) * 2 ** 9)


        w = int(w)


        x = (abs(x) & 0x3FF)
        y = (abs(y) & 0x3FF) << 10
        z = (abs(z) & 0x3FF) << 20
        w = (abs(w) & 0x3) << 30

        v = x | y | z | w
        return pack("<I", v)

    @staticmethod
    def packVertexStorageType(value,storageType):
        ST = StreamData.VertexElementDesc.StorageType
        r = BytePacker
        if storageType == ST.Undefined:
            raise Exception("Undefined Storage Type")
        if storageType == ST.SignedShortNormalized:
            return r.int16Norm(value)
        if storageType == ST.Float:
            return r.float(value)
        if storageType == ST.HalfFloat:
            return r.float16(value)
        if storageType == ST.UnsignedByteNormalized:
            return r.uint8Norm(value)
        if storageType == ST.SignedShort:
            return r.int8(value)
        if storageType == ST.X10Y10Z10W2Normalized:
            if len(value) == 3:
                return r.X10Y10Z10W2(value[0], value[1], value[2], 1.0)
            elif len(value) == 4:
                return r.X10Y10Z10W2(value[0],value[1],value[2],value[3])
            else:
                raise Exception("Unexpected value in X10Y10Z10W2 packing.")
        if storageType == ST.UnsignedByte:
            return r.uint8(value)
        if storageType == ST.UnsignedShort:
            return r.uint16(value)
        if storageType == ST.UnsignedShortNormalized:
            return r.uint16Norm(value)
        if storageType == ST.UNorm8sRGB:
            raise Exception("Storage Type not handled")
        if storageType == ST.X10Y10Z10W2UNorm:
            raise Exception("Storage Type not handled")

def FillChunk(f):
    offset = f.tell()
    remainder = abs(256-offset%256)
    if remainder == 256:
        remainder = 0
    bRem = b'\x00' * remainder
    f.write(bRem)

def CopyFile(read,write,offset,size,buffersize=500000):
    read.seek(offset)
    chunks = size // buffersize
    for o in range(chunks):
        write.write(read.read(buffersize))
    write.write(read.read(size%buffersize))

def Parse4x4Matrix(f):
    r = ByteReader
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    for i in range(4):
        for c in range(4):
            value = r.float(f)
            # if value > 100:
            #     print(f.tell(),value)
            if c == 0:
                row1.append(value)
            if c == 1:
                row2.append(value)
            if c == 2:
                row3.append(value)
            if c == 3:
                row4.append(value)
    # for i in range(4):
    #     row1.append(r.float(f))
    # for i in range(4):
    #     row2.append(r.float(f))
    # for i in range(4):
    #     row3.append(r.float(f))
    # for i in range(4):
    #     row4.append(r.float(f))
    # matrix = (row1,row2,row3,row4)
    # print(mathutils.Matrix((row1,row2,row3,row4)))
    matrix = mathutils.Matrix((row1,row2,row3,row4)).inverted()
    return matrix

class HZDSettings(bpy.types.PropertyGroup):
    AssetPath: bpy.props.StringProperty(name="Asset Path", subtype='FILE_PATH', update=ClearProperties,
                                      description="Path to a game skeletal mesh file \n e.g.: models/characters/humans/aloy/animation/parts/aloy_ct_a_lx.core")
    HZDPath: bpy.props.StringProperty(name="Mesh Core",subtype='FILE_PATH', update=ClearProperties,
                                      description="Path to the mesh .core file")
    HZDAbsPath : bpy.props.StringProperty()
    GamePath: bpy.props.StringProperty(name="Game Path", subtype='FILE_PATH', update=ClearProperties,
                                       description="Path to the folder containing HorizonZeroDawn.exe \n e.g.: C:\SteamLibrary\steamapps\common\Horizon Zero Dawn\ ")
    GameAbsPath: bpy.props.StringProperty()
    WorkPath: bpy.props.StringProperty(name="Workspace Path", subtype='FILE_PATH', update=ClearProperties,
                                       description="Path to the folder where you want files to be extracted.")
    WorkAbsPath: bpy.props.StringProperty()
    HZDSize: bpy.props.IntProperty()
    SkeletonPath: bpy.props.StringProperty(name="Skeleton Core",subtype='FILE_PATH', update=ClearProperties,
                                           description="Path to the skeleton .core file")
    SkeletonAbsPath : bpy.props.StringProperty()
    SkeletonName: bpy.props.StringProperty(name="Skeleton Name") #DEPRECATED

    ModelHelpersPath : bpy.props.StringProperty(name="Model Helpers",subtype='FILE_PATH', update=ClearProperties,
                                           description="Path to the robot_modelhelpers.core file. If not set the tool will try to extract it for you.\n"
                                                       "This is used for placing detachable parts on the robot skeleton.")

    NVTTPath : bpy.props.StringProperty(name="NVTT Path", default=str(NVTTDefaultPath) if os.path.exists(NVTTDefaultPath) else "", subtype='FILE_PATH',
                                        description="Path to the Nvidia Texture Tool nvtt_export.exe")

    LodDistance0: bpy.props.FloatProperty(name="Lod Distance 0")
    LodDistance1: bpy.props.FloatProperty(name="Lod Distance 1")
    LodDistance2: bpy.props.FloatProperty(name="Lod Distance 2")
    LodDistance3: bpy.props.FloatProperty(name="Lod Distance 3")
    LodDistance4: bpy.props.FloatProperty(name="Lod Distance 4")
    LodDistance5: bpy.props.FloatProperty(name="Lod Distance 5")
    LodDistance6: bpy.props.FloatProperty(name="Lod Distance 6")
    LodDistance7: bpy.props.FloatProperty(name="Lod Distance 7")
    LodDistance8: bpy.props.FloatProperty(name="Lod Distance 8")
    LodDistance9: bpy.props.FloatProperty(name="Lod Distance 9")
    LodDistance10: bpy.props.FloatProperty(name="Lod Distance 10")
    LodDistance11: bpy.props.FloatProperty(name="Lod Distance 11")
    LodDistance12: bpy.props.FloatProperty(name="Lod Distance 12")
    LodDistance13: bpy.props.FloatProperty(name="Lod Distance 13")
    LodDistance14: bpy.props.FloatProperty(name="Lod Distance 14")
    LodDistance15: bpy.props.FloatProperty(name="Lod Distance 15")
    ExtractTextures : bpy.props.BoolProperty(name="Extract Textures",default=False,description="Toggle the extraction of textures. When importing a mesh, texture will be detected and extracted to the Workspace directory.")
    OverwriteTextures : bpy.props.BoolProperty(name="Overwrite Textures",default=False,description="Overwrite existing textures, when textures will be extracted to the Workspace directory.")

def ParsePosition(f,storageType):
    r = ByteReader
    x = float(r.readVertexStorageType(f,storageType))
    y = float(r.readVertexStorageType(f, storageType))
    z = float(r.readVertexStorageType(f, storageType))
    return [x,y,z]
def ParseTris(f,is32bit=False):
    r = ByteReader
    if is32bit:
        return [r.uint32(f),r.uint32(f),r.uint32(f)]
    else:
        return [r.uint16(f),r.uint16(f),r.uint16(f)]
def ParseNormal(f,storageType):
    r = ByteReader
    if storageType == StreamData.VertexElementDesc.StorageType.X10Y10Z10W2Normalized:
        return r.readVertexStorageType(f,storageType)[0:3]
    x = float(r.readVertexStorageType(f,storageType))
    y = float(r.readVertexStorageType(f, storageType))
    z = float(r.readVertexStorageType(f, storageType))
    return [x,y,z]
def ParseUVChannel(f,storageType):
    r = ByteReader
    u = float(r.readVertexStorageType(f,storageType))
    v = float(r.readVertexStorageType(f,storageType))
    return [u,v]
def ParseBoneWeights(f,vindex,streamData):
    r = ByteReader
    vs:StreamData = streamData
    et = StreamData.VertexElementDesc.ElementType
    boneIndices = []
    boneWeights = []
    indiceWeight = {}
    for ei in vs.elementInfo:
        if ei.elementType in (et.BlendIndices,et.BlendIndices2):
            f.seek(vs.streamAbsOffset + vindex * vs.stride + ei.offset)
            for i in range(ei.count):
                boneIndices.append(r.readVertexStorageType(f,ei.storageType))
        if ei.elementType in (et.BlendWeights,et.BlendWeights2):
            f.seek(vs.streamAbsOffset + vindex * vs.stride + ei.offset)
            for i in range(ei.count):
                boneWeights.append(float(r.readVertexStorageType(f,ei.storageType)))
    if len(boneWeights) == 0: #happens when mesh is fully weight to a bone
        boneWeights = [1.0] * len(boneIndices)
    if len(boneIndices) != len(boneWeights):
        #Add 0.0 to match lenghts. In case there's more bone indices than weights.
        boneWeights.extend([0.0]*(len(boneIndices)-len(boneWeights)))
    #bones with 0.0 weight use the remaining weight
    remainderWeight = 1.0 - sum(boneWeights)
    #For some reason, the bone weghts are all offset by 1, the remainder goes to the first bone index.
    if remainderWeight > 0:
        boneWeights.insert(0,remainderWeight)

    for i,bi in enumerate(boneIndices):
        if boneWeights[i] != 0:
            indiceWeight[bi] = boneWeights[i]
    return indiceWeight
def ParseVertexColor(f,storageType):
    r = ByteReader
    red = float(r.readVertexStorageType(f,storageType))
    green = float(r.readVertexStorageType(f, storageType))
    blue = float(r.readVertexStorageType(f, storageType))
    alpha = float(r.readVertexStorageType(f, storageType))
    return [red,green,blue,alpha]

def findHelperInFile(filePath, objectName):
    with open(filePath, 'rb') as h:
        r = ByteReader
        DataBlock(h, expectedID=BlockIDs["SkeletonHelpers"])
        r.hashtext(h)
        helperCount = r.uint32(h)
        for i in range(helperCount):
            matrix = Parse4x4Matrix(h)
            name = r.hashtext(h)
            parentIndex = r.uint32(h)
            if name[:-7] == objectName: #remove "_helper"
                return parentIndex, matrix
        return None, None

def ImportMesh(isLodMesh, resIndex, meshIndex, primIndex):
    r = ByteReader
    # print(isLodMesh,Index,LODIndex,BlockIndex)
    if isLodMesh:
        prim = asset.LodMeshResources[resIndex].meshList[meshIndex].primitives[primIndex]
        meshName = asset.LodMeshResources[resIndex].meshList[meshIndex].meshName
        meshType = asset.LodMeshResources[resIndex].meshBase.drawableCullInfo.meshType
        assetName = asset.LodMeshResources[resIndex].objectName
    else:
        prim = asset.MultiMeshResources[resIndex].meshList[meshIndex].primitives[primIndex]
        meshName = asset.MultiMeshResources[resIndex].meshList[meshIndex].meshName
        meshType = asset.MultiMeshResources[resIndex].meshBase.drawableCullInfo.meshType
        assetName = asset.MultiMeshResources[resIndex].objectName
    say("\nImporting : " + str(primIndex) + "_" + meshName)
    say(prim)

    if not prim.vertexBlock.inStream:
        print("Cannot import mesh from .core")
        return


    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath
    stream = core+".stream"

    #bpy.ops.wm.console_toggle()

    # CREATE COLLECTION TREE #####################
    lodCollection = bpy.context.scene.collection
    # if bpy.context.scene.collection.children.find(assetName) >= 0:
    #     assetCollection = bpy.context.scene.collection.children[assetName]
    # else:
    #     assetCollection = bpy.context.blend_data.collections.new(name=assetName)
    #     bpy.context.scene.collection.children.link(assetCollection)
    # if isLodMesh:
    #     # LOD Collection
    #     if assetCollection.children.find("LOD " + str(meshIndex)) >= 0:
    #         lodCollection = assetCollection.children["LOD " + str(meshIndex)]
    #     else:
    #         lodCollection = bpy.context.blend_data.collections.new(name="LOD " + str(meshIndex))
    #         assetCollection.children.link(lodCollection)
    # else:
    #     # MultiMesh Collection
    #     if assetCollection.children.find("Multi Mesh") >= 0:
    #         lodCollection = assetCollection.children["Multi Mesh"]
    #     else:
    #         lodCollection = bpy.context.blend_data.collections.new(name="Multi Mesh")
    #         assetCollection.children.link(lodCollection)


    mesh = bpy.data.meshes.new(meshName+"_MESH")
    obj = bpy.data.objects.new(str(primIndex) + "_" + meshName, mesh)
    lodCollection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)

    vb:VertexArrayResource = prim.vertexBlock
    vs:StreamData = vb.vertexStream
    ns:StreamData = vb.normalsStream
    us:StreamData = vb.uvStream
    ia:IndexArrayResource = prim.faceBlock
    ish:StreamHandle = ia.indexStream
    et = StreamData.VertexElementDesc.ElementType
    HZDBones = []

    if meshType == CullInfo.MeshType.RegularSkinnedMesh:
        # Create Skeleton or Find it #####################
        armature = None
        armatureHash = str(ArchiveManager.get_file_hash(asset.LodMeshResources[0].meshList[0].skeletonRef.externalFile+".core"))
        for o in bpy.context.scene.collection.all_objects:
            if type(o.data) == bpy.types.Armature:
                if o.data.name == armatureHash:
                    armature = o
        if armature is None:
            print("IMPORT MESH: Skeleton object not found")
            armature = CreateSkeleton()

        # Get every bone name because blender's bone indices don't match HZD
        with open(HZDEditor.SkeletonAbsPath, 'rb') as f:
            f.seek(28)
            sktNameSize = r.int32(f)
            f.seek(4, 1)
            sktName = r.string(f, sktNameSize)
            boneCount = r.int32(f)
            for b in range(boneCount):
                boneNameSize = r.int32(f)
                f.seek(4, 1)
                boneName = r.string(f, boneNameSize)
                f.seek(6, 1)
                HZDBones.append(boneName)
        # Create vertex Groups
        for bone in HZDBones:
            obj.vertex_groups.new(name=bone)
        # Attach to Armature ##################
        obj.modifiers.new(name='Skeleton', type='ARMATURE')
        obj.modifiers['Skeleton'].object = armature
        obj.parent = armature
        # Check if armature is in the asset collection
        # if assetCollection.objects.find(armature.name) == -1:
        #     armature.users_collection[0].objects.unlink(armature)
        #     assetCollection.objects.link(armature)


    if vb.inStream:
        dataFile = stream
    else:
        dataFile = core
    print("\nOpening: ",dataFile)
    # Get the Position of Vertices from VertexStream
    with open(dataFile, 'rb') as f:
        for ei in vs.elementInfo:
            if ei.elementType == et.Pos: # POSITION
                print("Importing Vertex Positions")
                for v in range(vb.vertexCount):
                    f.seek(vs.streamAbsOffset + v * vs.stride + ei.offset)
                    # print(vs.streamAbsOffset,v,vs.stride,ei.offset)
                    # print(vs.streamAbsOffset + v * vs.stride + ei.offset)
                    # print(f.tell())
                    xyz = ParsePosition(f,ei.storageType)
                    bmv = bm.verts.new()
                    bmv.co = xyz

    # Get the Triangles from IndexArrayResource
    bm.verts.ensure_lookup_table()
    if ia.inStream:
        dataFile = stream
    else:
        dataFile = core
    if ia.indexFormat == 0:
        indexSize = 2
    else:
        indexSize = 4
    print("\nOpening: ", dataFile)
    with open(dataFile, 'rb') as f:
        print("Importing Faces")
        for i in range(0,ia.indexCount,3):
            f.seek(i*indexSize+ish.absOffset)
            tris = ParseTris(f,ia.indexFormat)
            if tris[0] != tris[1] != tris[2] != tris[0]:
                fvs = []
                for ti in tris:
                    fv = bm.verts[ti]
                    fvs.append(fv)
                bface = bm.faces.new(fvs)
                bface.smooth = True

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    #now that we have vertices and faces, we can add other kinds of data.
    if vb.inStream:
        dataFile = stream
    else:
        dataFile = core
    print("\nOpening: ",dataFile)

    with open(dataFile, 'rb') as f:
        if ns: # NORMALS STREAM
            bm = bmesh.new()
            bm.from_mesh(mesh)
            print(" Checking Normals Stream for more data")
            for ei in ns.elementInfo:
                if ei.elementType == et.Normal:
                    print("     Importing Normals")
                    normals = []
                    for v in range(vb.vertexCount):
                        f.seek(ns.streamAbsOffset + v * ns.stride + ei.offset)
                        n = ParseNormal(f,ei.storageType)
                        # print(n)
                        normals.append(n)
                    bm.to_mesh(mesh)
                    bm.free()
                    mesh.update()
                    mesh.use_auto_smooth = True
                    mesh.normals_split_custom_set_from_vertices(normals)
                    bm = bmesh.new()
                    bm.from_mesh(mesh)
                # elif ei.elementType == et.Tangent:
                #     print("Importing Tangent")
                # elif ei.elementType == et.TangentBFlip:
                #     print("Importing TangentBFlip")
                # elif ei.elementType == et.Binormal:
                #     print("Importing Binormal")
                else:
                    print("     ElementType not supported: ",ei.elementType)
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()

        # UV STREAM
        bm = bmesh.new()
        bm.from_mesh(mesh)
        print(" Checking UV Stream for more data")
        for ei in us.elementInfo:
            if ei.elementType in (et.UV0, et.UV1, et.UV2, et.UV3, et.UV4, et.UV5, et.UV6):
                print("     Importing UV Layer: ",ei.elementType.name)
                uv_layer = bm.loops.layers.uv.new(ei.elementType.name)
                for finder,face in enumerate(bm.faces):
                    for lindex,loop in enumerate(face.loops):
                        vindex = loop.vert.index
                        f.seek(us.streamAbsOffset+us.stride*vindex+ei.offset)
                        loop[uv_layer].uv = ParseUVChannel(f,ei.storageType)
            elif ei.elementType == et.Color:
                print("     Importing Vertex Color")
                color_layer = bm.loops.layers.color.new("Color")
                for finder,face in enumerate(bm.faces):
                    for lider, loop in enumerate(face.loops):
                        vindex = loop.vert.index
                        f.seek(us.streamAbsOffset+us.stride*vindex+ei.offset)
                        loop[color_layer] = ParseVertexColor(f,ei.storageType)
            else:
                print("     ElementType not supported: ",ei.elementType)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        # VERTEX STREAM
        bm = bmesh.new()
        bm.from_mesh(mesh)
        print(" Checking Vertex Stream for more data")
        for ei in vs.elementInfo:
            if ei.elementType == et.BlendIndices:
                print("     Importing BlendIndices")
                bm.to_mesh(mesh)
                bm.free()

                for v in range(vb.vertexCount):
                    indicesWeight = ParseBoneWeights(f,v,vs)
                    for bi in indicesWeight.keys():
                        vgName = HZDBones[bi]
                        # print(vgName,v,indicesWeight[bi])
                        obj.vertex_groups[vgName].add([v], indicesWeight[bi], "ADD")
                bm = bmesh.new()
                bm.from_mesh(mesh)
            elif ei.elementType in (et.BlendIndices2,et.BlendWeights,et.BlendWeights2):
                pass #Handled by BlendIndices ParseBoneWeights()
            elif ei.elementType == et.Pos:
                pass #Handled earlier in this function
            else:
                print("     ElementType not supported: ",ei.elementType)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    if meshType == CullInfo.MeshType.StaticMesh and vb.inStream:
        # Get robot_modelhelpers.core path and skeleton path
        modelHelperPath = ""

        streamPath = vs.streamInfo.path
        streamPath = streamPath[6:]  # remove "cache:"
        streamPath = streamPath.rpartition("/")[0]  # remove file name and last /
        streamPath = streamPath[:-5]  # remove "parts"
        skeletonAssetPath = streamPath + "skeletons/mesh_skeleton_rootbone.core"
        modelHelperPath = streamPath + "robot_modelhelpers.core"
        print(modelHelperPath)
        print(skeletonAssetPath)

        if HZDEditor.ModelHelpersPath == "":
            #there's no reference to the core file of the asset itself, so we can't know what its path is.
            if modelHelperPath != "":
                AM = ArchiveManager()
                modelHelperPath = AM.FindAndExtract(modelHelperPath, False)
                HZDEditor.ModelHelpersPath = modelHelperPath
        else:
            modelHelperPath = bpy.path.abspath(HZDEditor.ModelHelpersPath)

        armatureHash = str(ArchiveManager.get_file_hash(skeletonAssetPath))
        print(skeletonAssetPath,armatureHash)
        armature = None
        for o in bpy.context.scene.collection.all_objects:
            if type(o.data) == bpy.types.Armature:
                if o.data.name == armatureHash:
                    armature = o

        if modelHelperPath != "" and armature is not None:
            parentIndex, matrix = findHelperInFile(modelHelperPath,assetName)

            if parentIndex is None:
                print("Model not found in robot_modelhelpers.core")
            else:

                HZDBones = []
                # Get every bone name because blender's bone indices don't match HZD
                AM = ArchiveManager()
                HZDEditor.SkeletonAbsPath = AM.FindAndExtract(skeletonAssetPath, False)

                with open(HZDEditor.SkeletonAbsPath, 'rb') as f:
                    f.seek(28)
                    sktNameSize = r.int32(f)
                    f.seek(4, 1)
                    sktName = r.string(f, sktNameSize)
                    boneCount = r.int32(f)
                    for b in range(parentIndex+1):
                        boneNameSize = r.int32(f)
                        f.seek(4, 1)
                        boneName = r.string(f, boneNameSize)
                        f.seek(6, 1)
                        HZDBones.append(boneName)
                        if b == parentIndex:
                            break
                print("Armature = ",armature,"   Bone = ", boneName, "   Matrix = ", matrix)
                obj.parent = armature
                obj.parent_type = 'BONE'
                obj.parent_bone = boneName
                obj.rotation_mode = 'QUATERNION'
                boneVector = mathutils.Vector([0.0,armature.data.bones[boneName].length,0.0])
                matrix = mathutils.Matrix.inverted_safe(matrix)

                obj.matrix_basis = matrix
                obj.location += -boneVector

        else:
            print("WARNING: Armature not found")

    if HZDEditor.ExtractTextures:
        if isLodMesh:
            #TODO need a better way to find that material. LodMeshResource possibly doesn't exist, material is possibly inside Primitive instead of mesh.
            matblock = asset.LodMeshResources[resIndex].meshList[meshIndex].materials[primIndex]
        else:
            matblock = asset.MultiMeshResources[resIndex].meshList[meshIndex].materials[primIndex]
        CreateMaterial(obj,matblock,meshName)
    #bpy.ops.wm.console_toggle()

def ExtractAsset(assetPath):

    AM = ArchiveManager()
    #Extract Asset

    filePath = AM.FindAndExtract(assetPath,False)
    fileStreamPath = AM.FindAndExtract(assetPath+".stream", False)

    HZDEditor = bpy.context.scene.HZDEditor
    HZDEditor.HZDPath = filePath

    ReadCoreFile()
    skeletonFile = asset.LodMeshResources[0].meshList[0].skeletonRef.externalFile #TODO should do more checks on here (not sure if static or skinned, not sure if external
    say(skeletonFile)

    fileSkeletonPath = AM.FindAndExtract(skeletonFile + ".core", False)
    HZDEditor.SkeletonPath = fileSkeletonPath
    return

def LoadNodeGroup(group_name):
    if bpy.data.node_groups.find(group_name) != -1:
        return(group_name)

    filepath = dict(node_hzdmt_cache())[group_name]
    with bpy.data.libraries.load(filepath, link=not(group_name.startswith("_"))) as (data_from, data_to):
        assert(group_name in data_from.node_groups)
        data_to.node_groups = [group_name]
    return(group_name)

def ExtractTexture(outWorkspace,texPath):
    texAs = None
    def BuildDDSHeader(tex:Texture) -> bytes:
        r = BytePacker
        data = bytes()
        flags = b'\x07\x10\x00\x00'
        data += flags

        data += r.uint32(tex.height * tex.arraySize if tex.type == Texture.TextureType['_2DARRAY'] else tex.height)
        data += r.uint32(tex.width)
        data += r.uint32(tex.thumbnailLength + tex.streamSize32)
        data += r.uint32(0)
        data += r.uint32(tex.mipCount + 1 if hasattr(tex, "mipCount") else 1)
        data += (b'\x00' * 4) * 11

        ddsPF = bytes()
        ddsPF += r.uint32(32)
        ddsPF += r.uint32(4)
        ddsPF += b'DX10'
        ddsPF += r.uint32(0)
        ddsPF += r.uint32(0)
        ddsPF += r.uint32(0)
        ddsPF += r.uint32(0)
        ddsPF += r.uint32(0)

        data += ddsPF
        data += r.uint32(0)
        data += r.uint32(0)
        data += r.uint32(0)
        data += r.uint32(0)
        data += r.uint32(0)

        data = b'DDS ' + r.uint32(len(data)+4) + data

        dx10 = b''
        assert tex.format in format_map, f"Unmapped image format: {tex.format.name}"
        dx10 += r.uint32(format_map[tex.format].value)
        dx10 += r.uint32(3)
        dx10 += r.uint32(0)
        dx10 += r.uint32(1)
        dx10 += r.uint32(0)

        data += dx10

        return data

    def ParseTexture(filePath):
        #Parse Extracted Texture core
        with open(filePath,'rb') as f:
            nonlocal texAs
            texAs = TextureAsset(f)

        outPath = pathlib.Path(filePath)
        #Extract Stream
        for t in texAs.textures:
            HZDEditor = bpy.context.scene.HZDEditor
            ddsImage = outPath.with_name(t.name + ".dds")
            if os.path.exists(HZDEditor.NVTTPath):
                outImage = outPath.with_name(t.name + ".png")
            else:
                outImage = ddsImage

            if outImage.exists() and not HZDEditor.OverwriteTextures:
                textureFiles.append(outImage)
            else:
                streamData = bytes()
                if t.streamSize32 > 0:
                    # Check if .stream was already there
                    if os.path.exists(outWorkspace + texPath+".core.stream"):
                        streamFilePath = outWorkspace + texPath+".core.stream"
                    else:
                        streamFilePath = AM.FindAndExtract(texPath,True)
                    with open(streamFilePath,'rb') as s:
                        s.seek(t.streamOffset)
                        streamData = s.read(t.streamSize64)
                        s.close()

                with ddsImage.open(mode='wb') as w:
                    w.write(BuildDDSHeader(t))
                    w.write(streamData)
                    w.write(t.thumbnail)
                    w.close()
                    if os.path.exists(HZDEditor.NVTTPath):
                        if any([platform.startswith(os_name) for os_name in ['linux', 'darwin', 'freebsd']]):
                            from os import environ
                            env = dict(os.environ)
                            env['LD_LIBRARY_PATH'] = str(HZDEditor.NVTTPath)[:-12]
                            subprocess.run([str(HZDEditor.NVTTPath), "-format", "png", str(ddsImage)], env=env)
                        else:
                            subprocess.run([str(HZDEditor.NVTTPath), str(ddsImage), "-o", str(outImage)])
                        if outImage.exists():
                            ddsImage.unlink(missing_ok=True)
                        else:
                            outImage = ddsImage
                    textureFiles.append(outImage)

    textureFiles = []
    AM = ArchiveManager()

    if os.path.exists(outWorkspace+texPath+".core"):
        ParseTexture(outWorkspace+texPath+".core")
    else:
        #Extract Core
        filePath = AM.FindAndExtract(texPath,False)

        ParseTexture(filePath)

    return textureFiles, texAs

def CreateMaterial(obj,matblock,meshName):
    HZDEditor = bpy.context.scene.HZDEditor

    UsageType_ValueMap = {"Invalid":"Float",  # 0
                 "Color":"Color",  # 1
                 "Alpha":"Float",  # 2
                 "Normal":"Vector",  # 3
                 "Reflectance":"Float",  # 4
                 "AO":"Float",  # 5
                 "Roughness":"Float",  # 6
                 "Height":"Float",  # 7
                 "Mask":"Color",  # 8
                 "Mask_Alpha":"Float",  # 9
                 "Incandescence":"Float",  # 10
                 "Translucency_Diffusion":"Float",  # 11
                 "Translucency_Amount":"Float",  # 12
                 "Misc_01":"Float",  # 13
                 "Count":"Float"}  # 14

    BaseTextureOutputs = {"Detail Strength": ("Float", 0.0),
                 "Fur Mask": ("Float", 0.0),
                 "Skin Mask": ("Float", 0.0),
                 "AO": ("Float", 1.0),
                 "Subsurface": ("Float", 0.0),
                 "Translucency": ("Float", 0.0),
                 "Reflectance": ("Float", 0.25),
                 "Roughness": ("Float", 0.5),
                 "Alpha": ("Float", 1.0),
                 "Normal": ("Vector", (0.0,0.0,1.0)),
                 "Normal Color": ("Color", (0.5,0.5,0.0,1.0)),
                 "Color": ("Color", (1.0,1.0,1.0,1.0)),
                 "CID Mask": ("Color", (0.0,0.0,0.0,1.0)),
                 "Array Mask": ("Float", 0.0)}

    BaseTextureBSDFLinks = ("AO",
                 "Subsurface",
                 "Translucency",
                 "Reflectance",
                 "Alpha")

    DetailTextureOutputs = {"Normal": ("Vector", (0.0,0.0,1.0)),
                 "Normal Color": ("Color", (0.5,0.5,0.0,1.0)),
                 "Color": ("Color", (0.5,0.5,0.5,1.0))}

    import hashlib
    baseTexName = "unknown"
    for t in matblock.uniqueTextures:
        if t.startswith('models'):
            baseTexName = t.split('/')
            baseTexName = baseTexName[len(baseTexName) - 1]
            break
    materialName = baseTexName + "_" +  hashlib.sha224(str(matblock.uniqueTextures).encode("utf8")).hexdigest()[:10]

    frameNodes = {}
    frame_y = {}
    if bpy.data.materials.find(materialName) == -1:
        # Create new material and replace 'Principled BSDF' with 'HZD BSDF'
        mat = bpy.data.materials.new(name=materialName)
        obj.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.remove(mat.node_tree.nodes.get('Principled BSDF'))
        matNode = mat.node_tree.nodes.get('Material Output')
        matNode.location = 900.0, 300.0

        bsdfNode = mat.node_tree.nodes.new('ShaderNodeGroup')
        bsdfNode.node_tree = bpy.data.node_groups[LoadNodeGroup('HZD BSDF')]
        bsdfNode.width = 250.0
        bsdfNode.location = 600.0, 300.0
        mat.node_tree.links.new(bsdfNode.outputs['BSDF'], matNode.inputs['Surface'])

        # Add 'Combine Textures' Node Group and connect with 'HZD BSDF'
        combineTexNode = mat.node_tree.nodes.new('ShaderNodeGroup')
        combineTexNode.node_tree = bpy.data.node_groups[LoadNodeGroup('Combine Textures')]
        combineTexNode.width = 200.0
        combineTexNode.location = 320.0, -40.0
        combineTexNode.inputs['Detail Roughness'].hide = True
        mat.node_tree.links.new(combineTexNode.outputs['Roughness'], bsdfNode.inputs['Roughness'])
        mat.node_tree.links.new(combineTexNode.outputs['Normal'], bsdfNode.inputs['Normal'])
        mat.node_tree.links.new(combineTexNode.outputs['Color'], bsdfNode.inputs['Base Color'])

        # Create Frame Node for texture grouping
        for i,fn in enumerate(('models', 'textures', 'shaders', 'shader_libraries')):
            frameNodes[fn] = mat.node_tree.nodes.new('NodeFrame')
            frameNodes[fn].label = fn
            frameNodes[fn].shrink = True
            frameNodes[fn].location = -350 if fn == "models" else -350 * i, 220.0 if fn == "models" else -380.0
            frame_y[fn] = -40.0

        texDetailMapArray = None

        # Iterate through textures
        for i,t in enumerate(matblock.uniqueTextures):
            images,texAsset = ExtractTexture(HZDEditor.WorkAbsPath, t)
            frameName = t.split('/')[0]
            if frameName == 'models':
                tiling = False
            else:
                tiling = True

            if texAsset.texSet is not None:

                groupName = t.split('/')
                groupName = groupName[len(groupName) - 1]
                groupHash = hashlib.sha224(t.encode("utf8")).hexdigest()[:10]
                if len(images) > 2 or groupName == baseTexName:
                    groupOutputs = BaseTextureOutputs
                else:
                    groupOutputs = DetailTextureOutputs
                print(frameName, groupName, groupHash)

                if bpy.data.node_groups.find(groupName) == -1:
                    # Create Node Group for texture set
                    offset_x = 250.0
                    offset_y = 0.0
                    texSetGroup = bpy.data.node_groups.new(groupName,"ShaderNodeTree")

                    # Outputs
                    texSetGroup_output = texSetGroup.nodes.new("NodeGroupOutput")
                    for out in groupOutputs:
                        texSetGroup.outputs.new("NodeSocket" + groupOutputs[out][0], out)
                        texSetGroup.outputs[out].default_value = groupOutputs[out][1]
                        texSetGroup_output.inputs[out].hide = True
                    texSetGroup_output.location = offset_x + 10.0, offset_y + 30.0

                    # Inputs
                    if tiling:
                        # Create 'Texture Coordinate' and 'Mapping' to scale textures with 'Tiling' input value
                        texSetGroup.inputs.new("NodeSocketFloat", "Tiling")
                        texSetGroup.inputs["Tiling"].default_value = 20.0
                        texSetGroup.inputs["Tiling"].min_value = 0.0
                        texSetGroup_input = texSetGroup.nodes.new("NodeGroupInput")
                        texSetGroup_input.location = offset_x - 900.0, offset_y - 250.0
                        coordNode = texSetGroup.nodes.new('ShaderNodeTexCoord')
                        coordNode.location = offset_x - 900.0, offset_y
                        mappingNode = texSetGroup.nodes.new('ShaderNodeMapping')
                        mappingNode.location = offset_x - 700.0, offset_y
                        texSetGroup.links.new(texSetGroup_input.outputs['Tiling'], mappingNode.inputs['Scale'])
                        texSetGroup.links.new(coordNode.outputs['UV'], mappingNode.inputs['Vector'])

                    for ii, setT in enumerate(texAsset.texSet.textures):
                        # Create Image Node
                        texName = groupName + "%02d" % ii
                        print("  " + texName)
                        for cha in setT.channelTypes:
                            print("    " + cha.usageType)
                        texNode = texSetGroup.nodes.new('ShaderNodeTexImage')
                        texNode.location = offset_x - 500.0, offset_y - 50.0 * ii
                        if tiling:
                            texSetGroup.links.new(mappingNode.outputs['Vector'], texNode.inputs['Vector'])
                        # Load image if not yet exists
                        texNode.image = bpy.data.images.get(images[ii].name, None)
                        if texNode.image is None:
                            texNode.image = bpy.data.images.load(str(images[ii]))
                        texNode.image.colorspace_settings.name = "Non-Color"
                        texNode.hide = True

                        # RGB CHANNEL OUTPUT
                        if all(cha.usageType == "Normal" for cha in setT.channelTypes[0:2]):
                            # Normal Map
                            normalConverter = texSetGroup.nodes.new("ShaderNodeGroup")
                            normalConverter.node_tree = bpy.data.node_groups[LoadNodeGroup("HZD Normal Map Converter")]
                            normalConverter.location = offset_x - 200.0, texNode.location[1] if tiling else offset_y - 50 * len(images)
                            normalConverter.hide = True
                            normalConverter.inputs['Strength'].hide = True
                            texSetGroup.links.new(texNode.outputs['Color'], normalConverter.inputs['Color'])
                            texSetGroup.links.new(normalConverter.outputs['Normal'], texSetGroup_output.inputs['Normal'])
                            texSetGroup.links.new(normalConverter.outputs['Normal Color'], texSetGroup_output.inputs['Normal Color'])
                            if not setT.channelTypes[2].usageType == "Normal" and setT.channelTypes[2].usageType != "Invalid":
                                # Blue channel
                                sepRGBNode = texSetGroup.nodes.new("ShaderNodeSeparateRGB")
                                if tiling:
                                    normalConverter.location = offset_x - 200.0, texNode.location[1] + 20.0
                                    sepRGBNode.location = offset_x - 200.0, texNode.location[1] - 20.0
                                else:
                                    sepRGBNode.location = offset_x - 200.0, texNode.location[1]
                                sepRGBNode.label = "Separate B"
                                sepRGBNode.hide = True
                                sepRGBNode.outputs['R'].hide = True
                                sepRGBNode.outputs['G'].hide = True
                                if not texSetGroup.outputs.get(setT.channelTypes[2].usageType):
                                    texSetGroup.outputs.new('NodeSocketFloat', setT.channelTypes[2].usageType)
                                texSetGroup.links.new(texNode.outputs['Color'], sepRGBNode.inputs['Image'])
                                texSetGroup.links.new(sepRGBNode.outputs['B'], texSetGroup_output.inputs[setT.channelTypes[2].usageType])

                        elif all(cha.usageType == setT.channelTypes[0].usageType for cha in setT.channelTypes[0:3]):
                            # no need to break the color
                            if setT.channelTypes[0].usageType == "Color":
                                texNode.image.colorspace_settings.name = "Non-Color" if groupOutputs == DetailTextureOutputs else "sRGB"
                            if setT.channelTypes[0].usageType == "Misc_01":
                                # Separate Misc_01
                                separateMisc = texSetGroup.nodes.new("ShaderNodeGroup")
                                separateMisc.node_tree = bpy.data.node_groups[LoadNodeGroup("Separate Misc_01")]
                                separateMisc.location = offset_x - 200.0, offset_y + 40.0
                                separateMisc.hide = True
                                texSetGroup.links.new(texNode.outputs['Color'], separateMisc.inputs['Color'])
                                for out in separateMisc.outputs:
                                  texSetGroup.links.new(separateMisc.outputs[out.name], texSetGroup_output.inputs[out.name])
                            else:
                                outputType = UsageType_ValueMap[setT.channelTypes[0].usageType]
                                if setT.channelTypes[0].usageType == "Mask" and groupOutputs == BaseTextureOutputs:
                                    setT.channelTypes[0].usageType = "CID Mask"
                                if not texSetGroup.outputs.get(setT.channelTypes[0].usageType):
                                    texSetGroup.outputs.new("NodeSocket"+outputType, setT.channelTypes[0].usageType)
                                texSetGroup.links.new(texNode.outputs['Color'], texSetGroup_output.inputs[setT.channelTypes[0].usageType])

                        else:
                            sepRGBNode = texSetGroup.nodes.new("ShaderNodeSeparateRGB")
                            sepRGBNode.location = offset_x - 200.0, texNode.location[1]
                            sepRGBNode.label = "Separate "
                            sepRGBNode.hide = True
                            texSetGroup.links.new(texNode.outputs['Color'], sepRGBNode.inputs['Image'])
                            for ic,cha in enumerate(setT.channelTypes[0:3]):
                                sepRGBNode.outputs[ic].hide = True
                                if cha.usageType == "Invalid":
                                    pass
                                else:
                                    # we gotta separate RGB
                                    if groupOutputs == BaseTextureOutputs and texDetailMapArray != None:
                                        if cha.usageType == "Mask":
                                            cha.usageType = "Array Mask"
                                        if cha.usageType == "Mask_Alpha":
                                            cha.usageType = "Detail Strength"
                                    sepRGBNode.label += "RGB"[ic]
                                    if not texSetGroup.outputs.get(cha.usageType):
                                        texSetGroup.outputs.new("NodeSocketFloat", cha.usageType)
                                    texSetGroup.links.new(sepRGBNode.outputs[ic], texSetGroup_output.inputs[cha.usageType])

                        # ALPHA CHANNEL OUTPUT
                        if setT.channelTypes[3].usageType in ("Invalid","Normal","Mask","Reflectance","Misc_01"):
                            pass
                        else:
                            if setT.channelTypes[3].usageType == "Mask_Alpha" and groupOutputs == BaseTextureOutputs and texDetailMapArray != None:
                                usageType = "Detail Strength"
                            else:
                                usageType = setT.channelTypes[3].usageType
                            if not texSetGroup.outputs.get(usageType):
                                texSetGroup.outputs.new("NodeSocketFloat", usageType)
                            texSetGroup.links.new(texNode.outputs['Alpha'], texSetGroup_output.inputs[usageType])

                        if any(cha.usageType == "Alpha" for cha in setT.channelTypes):
                            mat.blend_method = "BLEND"
                            print("enable Alpha Blend for Eevee")

                # Add texture set Node Group
                shaderGroup = mat.node_tree.nodes.new("ShaderNodeGroup")
                shaderGroup.parent = frameNodes[frameName]
                shaderGroup.node_tree = bpy.data.node_groups[groupName]
                shaderGroup.location = 0.0, frame_y[frameName]
                shaderGroup.width = 200.0
                shaderGroup.hide = (frameName != "models")
                frame_y[frameName] -= 50.0 - 450.0*(frameName == "models")

                # Link BaseTexture to BSDF
                texSetGroup_output = shaderGroup.node_tree.nodes['Group Output']
                for out in texSetGroup_output.inputs:
                    if out.name and not out.is_linked:
                        shaderGroup.outputs[out.name].hide = True
                if groupOutputs == BaseTextureOutputs:
                    for out in BaseTextureBSDFLinks:
                        if texSetGroup_output.inputs[out].is_linked:
                            mat.node_tree.links.new(shaderGroup.outputs[out], bsdfNode.inputs[out])
                    if texSetGroup_output.inputs['Detail Strength'].is_linked:
                        mat.node_tree.links.new(shaderGroup.outputs['Detail Strength'], combineTexNode.inputs['Fac'])
                    mat.node_tree.links.new(shaderGroup.outputs['Roughness'], combineTexNode.inputs['Base Roughness'])
                    mat.node_tree.links.new(shaderGroup.outputs['Normal Color'], combineTexNode.inputs['Base Normal'])
                    mat.node_tree.links.new(shaderGroup.outputs['Color'], combineTexNode.inputs['Base Color'])
                    if texSetGroup_output.inputs['Array Mask'].is_linked and texDetailMapArray != None:
                        mat.node_tree.links.new(shaderGroup.outputs['Array Mask'], texDetailMapArray.inputs['Mask'])
                    elif texSetGroup_output.inputs['CID Mask'].is_linked or texSetGroup_output.inputs['Skin Mask'].is_linked:
                    # Add 'Combine Detail Textures' Node Group and connect with 'Combine Textures'
                        combineDetailNode = mat.node_tree.nodes.new('ShaderNodeGroup')
                        combineDetailNode.node_tree = bpy.data.node_groups[LoadNodeGroup('Combine Detail Textures')]
                        combineDetailNode.width = 250.0
                        combineDetailNode.location = -10.0, -250.0
                        mat.node_tree.links.new(combineDetailNode.outputs['Normal Color'], combineTexNode.inputs['Detail Normal'])
                        mat.node_tree.links.new(combineDetailNode.outputs['Detail Color'], combineTexNode.inputs['Detail Color'])
                    if texSetGroup_output.inputs['CID Mask'].is_linked:
                        mat.node_tree.links.new(shaderGroup.outputs['CID Mask'], combineDetailNode.inputs['CID Mask'])
                    if texSetGroup_output.inputs['Skin Mask'].is_linked:
                        mat.node_tree.links.new(shaderGroup.outputs['Color'], bsdfNode.inputs['Subsurface Color'])
                        mat.node_tree.links.new(shaderGroup.outputs['Skin Mask'], combineDetailNode.inputs['Skin Mask'])

            else:
                for ii, image in enumerate(images):
                    texName = t.split('/')
                    texName = texName[len(texName)-1]
                    texHash = hashlib.sha224(t.encode("utf8")).hexdigest()[:10]
                    print(frameName, texName, texHash)
                    #pdb.set_trace()
                    if texAsset.textures[0].type == Texture.TextureType['_2DARRAY']:
                        texDetailMapArray = mat.node_tree.nodes.new("ShaderNodeGroup")
                        if bpy.data.node_groups.find(texName) == -1:
                            texDetailMapArray.node_tree = bpy.data.node_groups[LoadNodeGroup("_DetailMapArrayTemplate")]
                            texDetailMapArray.node_tree.name = texName
                        else:
                            texDetailMapArray.node_tree = bpy.data.node_groups[texName]
                        texDetailMapArray.width = 250.0
                        texDetailMapArray.location = -10.0, -250.0
                        #texDetailMapArray.parent = frameNodes[frameName]
                        #texDetailMapArray.location = 0.0, frame_y[frameName]
                        #texDetailMapArray.hide = True
                        texNode = texDetailMapArray.node_tree.nodes['Image Texture']
                        mat.node_tree.links.new(texDetailMapArray.outputs['Roughness'], combineTexNode.inputs['Detail Roughness'])
                        mat.node_tree.links.new(texDetailMapArray.outputs['Normal Color'], combineTexNode.inputs['Detail Normal'])
                        mat.node_tree.links.new(texDetailMapArray.outputs['Color'], combineTexNode.inputs['Detail Color'])
                    else:
                        # Create Image Node
                        texNode = mat.node_tree.nodes.new('ShaderNodeTexImage')
                        texNode.parent = frameNodes[frameName]
                        texNode.location = 0.0, frame_y[frameName]
                        frame_y[frameName] -= 50.0

                    texNode.image = bpy.data.images.get(image.name, None)
                    # Load image if not yet exists
                    if texNode.image is None:
                        texNode.image = bpy.data.images.load(str(image))
                    texNode.image.colorspace_settings.name = "Non-Color"
                    texNode.hide = True

    else:
        say("Material already exists")
        mat = bpy.data.materials[materialName]
        obj.data.materials.append(mat)

def CreateSkeleton():
    r = ByteReader
    HZDEditor = bpy.context.scene.HZDEditor
    SkeletonPath = HZDEditor.SkeletonAbsPath
    skeletonFile = ""
    if len(asset.LodMeshResources) > 0:
        if asset.LodMeshResources[0].meshList[0].skeletonRef.externalFile:
            skeletonFile = asset.LodMeshResources[0].meshList[0].skeletonRef.externalFile
            skeletonFile += ".core"
            print("Create Skeleton: Skeleton reference found -> ", skeletonFile)
        else:
            print("Create Skeleton: Could not find skeleton reference")
            return #failed to find skeleton path string for hash reference. The skeleton data wouldn't have a proper name.
    if SkeletonPath == "" and skeletonFile != "":
        print("Create Skeleton: Skeleton Path not specified, extracting from reference...")
        AM = ArchiveManager()
        fileSkeletonPath = AM.FindAndExtract(skeletonFile, False)
        HZDEditor.SkeletonPath = fileSkeletonPath
        SkeletonPath = HZDEditor.SkeletonAbsPath
        print("Create Skeleton: Skeleton Path = ",SkeletonPath)

    if SkeletonPath != "" and skeletonFile != "":
        print("Create Skeleton: Paths are valid, starting to create skeleton...")
        Bones = []
        ParentIndices = []

        with open(SkeletonPath,'rb') as f:
            f.seek(28)
            sktNameSize = r.int32(f)
            f.seek(4,1)
            sktName = r.string(f,sktNameSize)
            boneCount = r.int32(f)
            # print(boneCount)
            for b in range(boneCount):
                boneNameSize = r.int32(f)
                f.seek(4,1)
                boneName = r.string(f,boneNameSize)
                f.seek(4,1)
                parentIndex = r.int16(f)
                # print(parentIndex)
                Bones.append(boneName)
                ParentIndices.append(parentIndex)

        armatureName = str(ArchiveManager.get_file_hash(skeletonFile))
        armature = bpy.data.armatures.new(armatureName)
        obj = bpy.data.objects.new(sktName, armature)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        HZDEditor.SkeletonName = obj.name

        bpy.ops.object.mode_set(mode="EDIT")
        print("Create Skeleton: Creating Bones...")
        for i,b in enumerate(Bones):
            bone = armature.edit_bones.new(b)
            bone.parent = armature.edit_bones[ParentIndices[i]]
            # print(bone.parent)
            bone.tail = mathutils.Vector([0,0.0,0.1])
        # TODO Not every bone has a matrix, if it's not used by the asset that populated the BoneMatrices the bone will be at world origin
        print("Create Skeleton: Placing Bones...")
        for b in BoneMatrices:
            bone = armature.edit_bones[b]
            # bone.tail = mathutils.Vector([0,0,1])

            bone.transform(BoneMatrices[b])
            # bone.transform(mathutils.Matrix.Rotation(math.radians(-90),4,'X'))

        #TODO I can't figure out a better way to switch Z and Y axis
        print("Create Skeleton: Swaping Bones Z and Y axis")
        for b in armature.edit_bones:
            zaxis = b.z_axis
            length = 0.1 #default bone length

            if len(b.children) == 1:
                if b.children[0].head != mathutils.Vector([0.0,0.0,0.0]):
                    b.tail = b.children[0].head # connect bone to child
                else:
                    b.tail = b.head + (-zaxis * length) #unless the child is at 0,0,0
            else:
                # if no children or multiple children, make the bone point in the -z axis
                # this was necessary for static meshes attached to the skeleton
                b.tail = b.head + (-zaxis * length)

        bpy.ops.object.mode_set(mode='OBJECT')
        return obj
def UpdateSkeleton(armatureObject):
    obj = armatureObject
    armature = obj.data
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    #TODO update skeleton bone matrices when mesh is imported on an existing armature

def WritePosition(f,vertex,elementInfo):
    p = BytePacker
    ei:StreamData.VertexElementDesc = elementInfo
    x = vertex.co[0]
    y = vertex.co[1]
    z = vertex.co[2]
    w = 1.0
    if ei.count == 4:
        v = [x,y,z,w]
    elif ei.count == 3:
        v = [x,y,z]
    else:
        raise Exception("Position Format : Count of %d is not supported." % ei.count)
    if ei.storageType == StreamData.VertexElementDesc.StorageType.X10Y10Z10W2Normalized:
        f.write(p.packVertexStorageType(v,ei.storageType))
    else:
        for pos in v:
            f.write(p.packVertexStorageType(pos, ei.storageType))

def GetVertexBlendIndices(vertex):
    groupWeights = {}
    # Gather bone groups
    for vg in vertex.groups:
        if vg.weight > 0.0:
            groupWeights[vg.group] = vg.weight
    # Normalize
    totalWeight = 0
    for k in groupWeights.keys():
        totalWeight += groupWeights[k]
    if totalWeight == 0:
        raise Exception("Vertex {v} has no weight".format(v=vertex.index))
    normalizer = 1/totalWeight
    for gw in groupWeights:
        groupWeights[gw] *= normalizer
    #Sort Weights
    sortedWeights = sorted(groupWeights.items(),key=operator.itemgetter(1),reverse=True)
    #TruncateWeights
    truncWeights = sortedWeights[0:8] # 8 seems good, haven't seen meshes with more than 8 indices
    return truncWeights
def WriteBlendIndicesWeights(f,vertex,elementInfo):
    p = BytePacker
    ei: StreamData.VertexElementDesc = elementInfo
    et = StreamData.VertexElementDesc.ElementType
    vg = GetVertexBlendIndices(vertex) # At this point vg is a list of list instead of dict. ((58,0.4),(23,0.3))
    bi = b''
    if ei.elementType == et.BlendIndices:
        for i in range(ei.count):
            if i > len(vg)-1:
                bi += p.packVertexStorageType(vg[-1][0],ei.storageType) # fill with last index
            else:
                bi += p.packVertexStorageType(vg[i][0],ei.storageType)
    if ei.elementType == et.BlendIndices2:
        for i in range(ei.count):
            # I should be taking count of BlendIndice1 instead of 2. But I assume they both have same count.
            if i + ei.count > len(vg)-1:
                bi += p.packVertexStorageType(vg[-1][0],ei.storageType) # fill with last index
            else:
                bi += p.packVertexStorageType(vg[i + ei.count][0],ei.storageType)
    if ei.elementType == et.BlendWeights:
        for i in range(ei.count):
            if i + 1 > len(vg)-1:
                bi += p.packVertexStorageType(0.0,ei.storageType) # fill with 0.0
            else:
                bi += p.packVertexStorageType(vg[i+1][1],ei.storageType)
    if ei.elementType == et.BlendWeights2:
        for i in range(ei.count):
            if i + 1 + ei.count > len(vg)-1: # + 1 because we skipped first index before
                bi += p.packVertexStorageType(0.0,ei.storageType) # fill with 0.0
            else:
                bi += p.packVertexStorageType(vg[i + 1 + ei.count][1],ei.storageType)
    f.write(bi)

def WriteNormal(f,normal,elementInfo):
    p = BytePacker
    ei: StreamData.VertexElementDesc = elementInfo
    if ei.storageType == StreamData.VertexElementDesc.StorageType.X10Y10Z10W2Normalized:
        f.write(p.packVertexStorageType(normal,ei.storageType))
    else:
        for n in normal:
            f.write(p.packVertexStorageType(n,ei.storageType))
def WriteTangentBFlip(f,tangent,flip,elementInfo):
    p = BytePacker
    ei: StreamData.VertexElementDesc = elementInfo
    tf = []
    for t in tangent:
        tf.append(t)
    tf.append(flip)
    if ei.storageType == StreamData.VertexElementDesc.StorageType.X10Y10Z10W2Normalized:
        f.write(p.packVertexStorageType(tf,ei.storageType))
    else:
        for t in tf:
            f.write(p.packVertexStorageType(t,ei.storageType))

def GetUVs(editedMesh,uvIndex=0):
    UVs = [(0.0, 0.0)] * len(editedMesh.vertices)
    bm = bmesh.new()
    bm.from_mesh(editedMesh)
    bm.faces.ensure_lookup_table()
    # Get UVs and Color
    for bface in bm.faces:
        for loop in bface.loops:
            u = loop[bm.loops.layers.uv[uvIndex]].uv[0]
            v = loop[bm.loops.layers.uv[uvIndex]].uv[1]
            UVs[loop.vert.index] = [u,v]
    bm.to_mesh(editedMesh)
    bm.free()
    editedMesh.update()
    return UVs
def WriteUVs(f,UVs,index,elementInfo):
    p = BytePacker
    ei: StreamData.VertexElementDesc = elementInfo
    uv = b''
    for u in UVs[index]:
        uv += p.packVertexStorageType(u,ei.storageType)
    f.write(uv)

def GetColor(editedMesh):
    Colors = [(0,0,0,0)] * len(editedMesh.vertices)
    bm = bmesh.new()
    bm.from_mesh(editedMesh)
    bm.faces.ensure_lookup_table()
    # Get UVs and Color
    for bface in bm.faces:
        for loop in bface.loops:
            uv = loop[bm.loops.layers.color[0]].uv
            Colors[loop.vert.index] = uv
    return Colors
def WriteColor(f,vertexColors,index,elementInfo):
    p = BytePacker
    ei: StreamData.VertexElementDesc = elementInfo
    vc = b''
    for c in vertexColors[index]:
        vc += p.packVertexStorageType(c,ei.storageType)
    f.write(vc)

def WriteTriangle(f,poly,indexArray):
    p = BytePacker
    ia: IndexArrayResource = indexArray
    if ia.indexFormat:
        for v in poly.vertices:
            f.write(p.uint32(v))
    else:
        for v in poly.vertices:
            f.write(p.uint16(v))

def ExportMesh(isLodMesh, resIndex, meshIndex, primIndex):
    r = ByteReader
    p = BytePacker

    if isLodMesh:
        mesh = asset.LodMeshResources[resIndex].meshList[meshIndex]
        prim = mesh.primitives[primIndex]
        meshName = mesh.meshName
    else:
        mesh = asset.MultiMeshResources[resIndex].meshList[meshIndex]
        prim = mesh.primitives[primIndex]
        meshName = mesh.meshName
    print("\nExporting : " + str(primIndex) + "_" + meshName)

    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath
    stream = core + ".stream"

    vb: VertexArrayResource = prim.vertexBlock
    vs: StreamData = vb.vertexStream
    ns: StreamData = vb.normalsStream
    us: StreamData = vb.uvStream
    ia: IndexArrayResource = prim.faceBlock
    ish: StreamHandle = ia.indexStream
    et = StreamData.VertexElementDesc.ElementType

    objectName = str(primIndex) + "_" + meshName
    editedMesh = bpy.data.objects[objectName].data
    boneCount = len(bpy.data.objects[HZDEditor.SkeletonName].data.bones)

    # Check if there's already a modded file
    # if os.path.exists(core + "MOD"):
    #     sourcecore = core + "MOD"
    #     if os.path.exists(stream + "MOD"):
    #         sourcestream = stream + "MOD"
    #     else:
    #         raise Exception("Modded Core but no Modded Stream")
    # else:
    #     sourcecore = core
    #     sourcestream = stream

    sourcecore = core
    sourcestream = stream
    coresize = os.path.getsize(sourcecore)
    streamsize = os.path.getsize(sourcestream)

    # Write Stream
    with open(sourcestream, 'rb') as f, open(stream + "TMP", 'wb+') as w:
        CopyFile(f, w, 0,vs.streamAbsOffset) #Copy the source file up to the vertex position

        #Write Vertex Stream
        newVertexStreamOffset = w.tell()
        for v in editedMesh.vertices:
            for ei in vs.elementInfo:
                if ei.elementType == et.Pos:
                    WritePosition(w,v,ei)
                elif ei.elementType in (et.BlendIndices,et.BlendIndices2,et.BlendWeights,et.BlendWeights2):
                    WriteBlendIndicesWeights(w,v,ei)
                else:
                    print("     ElementType not supported in Vertex Stream: ", ei.elementType)
        FillChunk(w)
        newVertexStreamSize = w.tell() - newVertexStreamOffset

        if ns:
            # Write Normals Stream
            NTB = [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0)] * len(editedMesh.vertices)  # Normal Tangent Bi-tangent
            # Get Normals
            editedMesh.loops.data.calc_tangents()
            for l in editedMesh.loops:
                if l.bitangent_sign == -1:
                    flip = 1.0
                else:
                    flip = 0.0
                NTB[l.vertex_index] = (l.normal, l.tangent, flip)

            newNormalStreamOffset = w.tell()
            for v in editedMesh.vertices:
                for ei in ns.elementInfo:
                    if ei.elementType == et.Normal:
                        WriteNormal(w,NTB[v.index][0],ei)
                    elif ei.elementType == et.TangentBFlip:
                        WriteTangentBFlip(w,NTB[v.index][1],NTB[v.index][2],ei)
                    else:
                        print("     ElementType not supported in Normals Stream: ", ei.elementType)
            FillChunk(w)
            newNormalStreamSize = w.tell() - newNormalStreamOffset

        # Write UV per vertex
        newUVStreamOffset = w.tell()
        # Get All UVs
        allUVs = []


        for ei in us.elementInfo:
            if ei.elementType in (et.UV0, et.UV1, et.UV2, et.UV3, et.UV4, et.UV5, et.UV6):
                allUVs.append(GetUVs(editedMesh,int(ei.elementType.name[-1])))
            if ei.elementType == et.Color:
                vertexColor = GetColor(editedMesh)

        for v in editedMesh.vertices:
            for ei in us.elementInfo:
                if ei.elementType in (et.UV0,et.UV1,et.UV2,et.UV3,et.UV4,et.UV5,et.UV6):
                    WriteUVs(w,allUVs[int(ei.elementType.name[-1])],v.index,ei)
                elif ei.elementType == et.Color:
                    WriteColor(w,vertexColor,v.index,ei)
                else:
                    print("     ElementType not supported in Normals Stream: ", ei.elementType)
        FillChunk(w)
        newUVStreamSize = w.tell() - newUVStreamOffset

        newIndexStreamOffset = w.tell()
        for poly in editedMesh.polygons:
            WriteTriangle(w,poly,ia)
        FillChunk(w)
        newIndexStreamSize = w.tell() - newIndexStreamOffset

        endOffset = ish.resourceOffset + ish.resourceLength
        CopyFile(f, w, endOffset, streamsize - endOffset)  # Copy the source file to the end.

    # Write Core
    with open(sourcecore, 'rb') as f, open(core+"TMP",'wb+') as w:
        CopyFile(f,w,0,coresize) #full copy of source core
        # WRITE NEW VALUES FOR THE CURRENT MESH BLOCK
        #Vertex Counts
        w.seek(vb.posVCount)
        w.write(p.int32(len(editedMesh.vertices)))
        w.seek(mesh.skinInfo.meshInfos[primIndex].posVCount)
        w.write(p.int32(len(editedMesh.vertices)))
        #Vertex
        w.seek(vs.streamInfo.offsetPos)
        w.write(p.uint64(newVertexStreamOffset))
        w.seek(vs.streamInfo.lengthPos)
        w.write(p.uint64(newVertexStreamSize))
        #Edges
        #TODO somehow gotta handle unknown DataBufferResources
        #Normals
        if ns:
            w.seek(ns.streamInfo.offsetPos)
            w.write(p.uint64(newVertexStreamOffset))
            w.seek(ns.streamInfo.lengthPos)
            w.write(p.uint64(newNormalStreamSize))
        #UVs
        w.seek(us.streamInfo.offsetPos)
        w.write(p.uint64(newVertexStreamOffset))
        w.seek(us.streamInfo.lengthPos)
        w.write(p.uint64(newUVStreamSize))
        #Faces
        w.seek(ia.posIndexCount)
        w.write(p.int32(len(editedMesh.polygons)*3))
        w.seek(prim.posIndexCount)
        w.write(p.int32(len(editedMesh.polygons)*3))
        w.seek(ish.offsetPos)
        w.write(p.uint64(newIndexStreamOffset))
        w.seek(ish.lengthPos)
        w.write(p.uint64(newIndexStreamSize))

        # the place where new mesh block ends minus where it ended before
        DiffOff = (newIndexStreamOffset + newIndexStreamSize) - (ish.resourceOffset + ish.resourceLength)
        print(DiffOff)
        # print(DiffOff)
        def AddDiff(pos,diff=DiffOff):
            if pos != 0:
                w.seek(pos)
                oldOffset = r.uint64(w)
                w.seek(pos)
                w.write(p.uint64(oldOffset + diff))
        def mdDiff(p_mesh, p_prim):
            # Vertex
            if p_prim.vertexBlock:
                AddDiff(p_prim.vertexBlock.vertexStream.streamInfo.offsetPos)
            # Edge
            if p_mesh.skinInfo.meshInfos[primIndex].edgeRef.type.name == "Internal":
                AddDiff(p_mesh.skinInfo.edgeData.dataOffset)
            # Normals
            if p_prim.vertexBlock.normalsStream:
                AddDiff(p_prim.vertexBlock.normalsStream.streamInfo.offsetPos)
            # UV
            if p_prim.vertexBlock.uvStream:
                AddDiff(p_prim.vertexBlock.uvStream.streamInfo.offsetPos)
            # Faces
            if p_prim.faceBlock:
                AddDiff(p_prim.faceBlock.indexStream.offsetPos)

        #Group are after Objects, so no need to add to objects.
        if isLodMesh:
            # Remaining primitives of current Lod
            for prim in asset.LodMeshResources[resIndex].meshList[meshIndex].primitives[primIndex + 1:]:
                mdDiff(mesh,prim)
            # The following LODs
            for l in asset.LodMeshResources[resIndex].meshList[meshIndex + 1:]:
                for prim in l.primitives:
                    mdDiff(l,prim)
            #just in case there are other groups
            for g in asset.LodMeshResources[resIndex + 1:]:
                for l in g.meshList:
                    for prim in l.primitives:
                        mdDiff(l,prim)

        else:
            # Remaining mesh blocks of current Lod
            for prim in asset.MultiMeshResources[resIndex].meshList[meshIndex].primitives[primIndex + 1:]:
                mdDiff(mesh,prim)
            # The following LODs
            for l in asset.MultiMeshResources[resIndex].meshList[meshIndex + 1:]:
                for prim in l.primitives:
                    mdDiff(l,prim)
            # the other objects
            for o in asset.MultiMeshResources[resIndex + 1:]:
                for l in o.meshList:
                    for prim in l.primitives:
                        mdDiff(l,prim)
            # do every md in every lod of every LODGroup
            for g in asset.LodMeshResources:
                for l in g.meshList:
                    for prim in l.primitives:
                        mdDiff(l,prim)
    # Delete Source Core
    # if os.path.exists(core + "MOD"):
    #     os.remove(core+"MOD")
    #     # Delete Source Stream
    #     if os.path.exists(stream + "MOD"):
    #         os.remove(stream + "MOD")

    #Rename Core and Stream
    # os.rename(core + "TMP", core + "MOD")
    # os.rename(stream + "TMP", stream + "MOD")
    os.remove(core)
    os.remove(stream)
    os.rename(core + "TMP", core)
    os.rename(stream + "TMP", stream)
    ReadCoreFile()

def SaveDistances(Index):
    p = BytePacker
    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath

    with open(core,'rb+') as w:
        w.seek(asset.LodMeshResources[Index].blockStartOffset)
        r = ByteReader
        w.seek(16, 1)
        r.hashtext(w)
        w.seek(24, 1)
        w.seek(8, 1)
        w.seek(4,1)
        w.seek(16, 1)
        LODCount = r.int32(w)
        for i in range(LODCount):
            w.seek(17, 1)
            w.write(p.float(HZDEditor["LodDistance" + str(i)]))



class Asset:
    def __init__(self):
        # these are lists just in case, but I haven't seen any asset with more than one of those.
        self.LodMeshResources = []
        self.MultiMeshResources = []
        self.RegularSkinnedMeshResources = []
        self.StaticMeshResources = []


BlockIDs = {"RegularSkinnedMeshResource" : 10982056603708398958,
        "VertexArrayResource" : 13522917709279820436,
        "SkinnedMeshBoneBindings" : 232082505300933932,
        "SkinnedMeshBoneBoundingBoxes" : 1425406424293942754,
        "RegularSkinnedMeshResourceSkinInfo" : 4980347625154103665,
        "RenderingPrimitiveResource" : 17523037150162385132,
        "IndexArrayResource" : 12198706699739407665,
        "RenderEffectResource" : 12029122079492233037,
        "LodMeshResource" : 6871768592993170868,
        "ShaderResource" : 5215210673454096253, #was 5215210673454096253 5561636305660569489
        "TextureResource" : 17501462827539052646,
        "MultiMeshResource" : 7022335006738406101,
        "DataBufferResource" : 10234768860597628846,
        "StaticMeshResource" : 17037430323200133752,
        "SKDTreeResource" : 13505794420212475061,
        "SkeletonHelpers" : 6306064744810253771
            }
asset = Asset()

class DataBlock:
    def __init__(self,f,expectedID=0,expectedGUID=0):
        r = ByteReader

        self.expectedID = expectedID
        self.ID = r.uint64(f)
        if expectedID != 0:
            if self.ID != self.expectedID:
                raise Exception("%s  --  Invalid Block ID: got %d expected %d"%(self.__class__.__name__,self.ID ,self.expectedID))
        self.size = r.int32(f)
        self.blockStartOffset = f.tell()
        self.guid = r.guid(f)
        if expectedGUID != 0:
            self.validateGUID(expectedGUID)
        # print(self.__class__.__name__)
        # print("ID = ",self.ID,"\n","Size = ",self.size,"\nStart = ",self.blockStartOffset)
    def validateGUID(self,expectedGUID):
        if self.guid == expectedGUID:
            return
        else:
            raise Exception("%s  --  Invalid Block GUID: got %s expected %s"%(self.__class__.__name__,str(self.guid.hex()) ,str(expectedGUID)))
    def EndBlock(self,f):
        f.seek(self.blockStartOffset + self.size)

class TextureAsset:
    def __init__(self,f):
        self.textures = []
        self.texSet = None
        r = ByteReader
        ID = r.uint64(f)
        f.seek(-8,1)
        if ID == 1009496109439982815: #Texture Set
            self.texSet = TextureSet(f)
            for tex in self.texSet.textures:
                if type(tex.textureRef) is str:
                    pass
                else:
                    self.textures.append(Texture(f))
        elif ID == 17501462827539052646: #Single Texture
            self.textures.append(Texture(f))
        else:
            raise Exception("That wasn't a texture asset.")
class TextureSet(DataBlock):
    class TextureUsage:
        UsageType = ["Invalid", # 0
                     "Color",   # 1
                     "Alpha",   # 2
                     "Normal",  # 3
                     "Reflectance",  # 4
                     "AO",      # 5
                     "Roughness",  # 6
                     "Height",  # 7
                     "Mask",    # 8
                     "Mask_Alpha",  # 9
                     "Incandescence", # 10
                     "Translucency_Diffusion", # 11
                     "Translucency_Amount", # 12
                     "Misc_01", # 13
                     "Count"] # 14
    class TextureDetails:
        class ChannelDetails:
            def __init__(self,f):
                r = ByteReader
                self.usageTypeIndex = r.uint8(f)
                self.usageType = TextureSet.TextureUsage.UsageType[self.usageTypeIndex & 0x0F]
            def __str__(self):
                return self.usageType

        def __init__(self,f):
            r = ByteReader
            f.seek(9,1)
            self.channelTypes = [self.ChannelDetails(f) for _ in range(4)]
            f.seek(4,1)
            refType = r.uint8(f)
            if refType > 0:
                self.textureRef = r.uuid(f)
            elif refType in [2,3]:
                self.textureRef = r.hashtext(f)

    def __init__(self,f):
        super().__init__(f)
        r = ByteReader

        self.name = r.hashtext(f)
        self.textureCount = r.int32(f)
        self.textures = [self.TextureDetails(f) for _ in range(self.textureCount)]

        self.EndBlock(f)
class Texture(DataBlock):
    class PixelFormat(IntEnum):
        INVALID = 0x4C,
        RGBA_5551 = 0x0,
        RGBA_5551_REV = 0x1,
        RGBA_4444 = 0x2,
        RGBA_4444_REV = 0x3,
        RGB_888_32 = 0x4,
        RGB_888_32_REV = 0x5,
        RGB_888 = 0x6,
        RGB_888_REV = 0x7,
        RGB_565 = 0x8,
        RGB_565_REV = 0x9,
        RGB_555 = 0xA,
        RGB_555_REV = 0xB,
        RGBA_8888 = 0xC,
        RGBA_8888_REV = 0xD,
        RGBE_REV = 0xE,
        RGBA_FLOAT_32 = 0xF,
        RGB_FLOAT_32 = 0x10,
        RG_FLOAT_32 = 0x11,
        R_FLOAT_32 = 0x12,
        RGBA_FLOAT_16 = 0x13,
        RGB_FLOAT_16 = 0x14,
        RG_FLOAT_16 = 0x15,
        R_FLOAT_16 = 0x16,
        RGBA_UNORM_32 = 0x17,
        RG_UNORM_32 = 0x18,
        R_UNORM_32 = 0x19,
        RGBA_UNORM_16 = 0x1A,
        RG_UNORM_16 = 0x1B,
        R_UNORM_16 = 0x1C,  # Old: INTENSITY_16
        RGBA_UNORM_8 = 0x1D,
        RG_UNORM_8 = 0x1E,
        R_UNORM_8 = 0x1F,  # Old: INTENSITY_8
        RGBA_NORM_32 = 0x20,
        RG_NORM_32 = 0x21,
        R_NORM_32 = 0x22,
        RGBA_NORM_16 = 0x23,
        RG_NORM_16 = 0x24,
        R_NORM_16 = 0x25,
        RGBA_NORM_8 = 0x26,
        RG_NORM_8 = 0x27,
        R_NORM_8 = 0x28,
        RGBA_UINT_32 = 0x29,
        RG_UINT_32 = 0x2A,
        R_UINT_32 = 0x2B,
        RGBA_UINT_16 = 0x2C,
        RG_UINT_16 = 0x2D,
        R_UINT_16 = 0x2E,
        RGBA_UINT_8 = 0x2F,
        RG_UINT_8 = 0x30,
        R_UINT_8 = 0x31,
        RGBA_INT_32 = 0x32,
        RG_INT_32 = 0x33,
        R_INT_32 = 0x34,
        RGBA_INT_16 = 0x35,
        RG_INT_16 = 0x36,
        R_INT_16 = 0x37,
        RGBA_INT_8 = 0x38,
        RG_INT_8 = 0x39,
        R_INT_8 = 0x3A,
        RGB_FLOAT_11_11_10 = 0x3B,
        RGBA_UNORM_10_10_10_2 = 0x3C,
        RGB_UNORM_11_11_10 = 0x3D,
        DEPTH_FLOAT_32_STENCIL_8 = 0x3E,
        DEPTH_FLOAT_32_STENCIL_0 = 0x3F,
        DEPTH_24_STENCIL_8 = 0x40,
        DEPTH_16_STENCIL_0 = 0x41,
        BC1 = 0x42,  # Old: S3TC1
        BC2 = 0x43,  # Old: S3TC3
        BC3 = 0x44,  # Old: S3TC5
        BC4U = 0x45,
        BC4S = 0x46,
        BC5U = 0x47,
        BC5S = 0x48,
        BC6U = 0x49,
        BC6S = 0x4A,
        BC7 = 0x4B
    class TextureType(IntEnum):
        _2D = 0x0,
        _3D = 0x1,
        CUBE_MAP = 0x2,
        _2DARRAY = 0x3
    def __init__(self,f):
        super().__init__(f)
        r = ByteReader
        self.name = r.hashtext(f)
        self.type : Texture.TextureType = self.TextureType(r.uint16(f))
        width = r.uint16(f)
        self.width = width & 0x3FFF
        height = r.uint16(f)
        self.height = height & 0x3FFF
        self.arraySize = r.uint16(f)
        f.seek(1,1)
        self.format : Texture.PixelFormat = self.PixelFormat(r.uint8(f))
        f.seek(2,1)
        f.seek(20,1)
        self.imageChunkSize= r.int32(f)
        self.thumbnailLength = r.int32(f)
        self.streamSize32 = r.int32(f)
        self.mipCount = r.int32(f)
        if self.streamSize32 > 0:
            self.streamPath = r.path(f) [6:] #remove "cache:"
            self.streamOffset = r.uint64(f)
            self.streamSize64 = r.uint64(f)
        else:
            padding = self.imageChunkSize - (self.thumbnailLength + 8)
            f.seek(padding,1)
        self.thumbnail = f.read(self.thumbnailLength)

        self.EndBlock(f)
format_map: Dict[Texture.PixelFormat, DXGI] = {
    Texture.PixelFormat.INVALID: DXGI.DXGI_FORMAT_UNKNOWN,
    # Texture.PixelFormat.RGBA_5551: ,
    # Texture.PixelFormat.RGBA_5551_REV: ,
    # Texture.PixelFormat.RGBA_4444: ,
    # Texture.PixelFormat.RGBA_4444_REV: ,
    # Texture.PixelFormat.RGB_888_32: ,
    # Texture.PixelFormat.RGB_888_32_REV: ,
    # Texture.PixelFormat.RGB_888: ,
    # Texture.PixelFormat.RGB_888_REV: ,
    # Texture.PixelFormat.RGB_565: ,
    # Texture.PixelFormat.RGB_565_REV: ,
    # Texture.PixelFormat.RGB_555: ,
    # Texture.PixelFormat.RGB_555_REV: ,
    Texture.PixelFormat.RGBA_8888: DXGI.DXGI_FORMAT_R8G8B8A8_TYPELESS,
    # Texture.PixelFormat.RGBA_8888_REV: ,
    # Texture.PixelFormat.RGBE_REV: ,
    Texture.PixelFormat.RGBA_FLOAT_32: DXGI.DXGI_FORMAT_R32G32B32A32_FLOAT,
    Texture.PixelFormat.RGB_FLOAT_32: DXGI.DXGI_FORMAT_R32G32B32_FLOAT,
    Texture.PixelFormat.RG_FLOAT_32: DXGI.DXGI_FORMAT_R32G32_FLOAT,
    Texture.PixelFormat.R_FLOAT_32: DXGI.DXGI_FORMAT_R32_FLOAT,
    Texture.PixelFormat.RGBA_FLOAT_16: DXGI.DXGI_FORMAT_R16G16B16A16_FLOAT,
    # Texture.PixelFormat.RGB_FLOAT_16: ,
    Texture.PixelFormat.RG_FLOAT_16: DXGI.DXGI_FORMAT_R16G16_FLOAT,
    Texture.PixelFormat.R_FLOAT_16: DXGI.DXGI_FORMAT_R16_FLOAT,
    # Texture.PixelFormat.RGBA_UNORM_32: ,
    # Texture.PixelFormat.RG_UNORM_32: ,
    # Texture.PixelFormat.R_UNORM_32: ,
    Texture.PixelFormat.RGBA_UNORM_16: DXGI.DXGI_FORMAT_R16G16B16A16_UNORM,
    Texture.PixelFormat.RG_UNORM_16: DXGI.DXGI_FORMAT_R16G16_UNORM,
    Texture.PixelFormat.R_UNORM_16: DXGI.DXGI_FORMAT_R16_UNORM,  # Old: INTENSITY_16
    Texture.PixelFormat.RGBA_UNORM_8: DXGI.DXGI_FORMAT_R8G8B8A8_UNORM,
    Texture.PixelFormat.RG_UNORM_8: DXGI.DXGI_FORMAT_R8G8_UNORM,
    Texture.PixelFormat.R_UNORM_8: DXGI.DXGI_FORMAT_R8_UNORM,  # Old: INTENSITY_8
    # Texture.PixelFormat.RGBA_NORM_32: ,
    # Texture.PixelFormat.RG_NORM_32: ,
    # Texture.PixelFormat.R_NORM_32: ,
    Texture.PixelFormat.RGBA_NORM_16: DXGI.DXGI_FORMAT_R16G16B16A16_SNORM,
    Texture.PixelFormat.RG_NORM_16: DXGI.DXGI_FORMAT_R16G16_SNORM,
    Texture.PixelFormat.R_NORM_16: DXGI.DXGI_FORMAT_R16_SNORM,
    Texture.PixelFormat.RGBA_NORM_8: DXGI.DXGI_FORMAT_R8G8B8A8_SNORM,
    Texture.PixelFormat.RG_NORM_8: DXGI.DXGI_FORMAT_R8G8_SNORM,
    Texture.PixelFormat.R_NORM_8: DXGI.DXGI_FORMAT_R8_SNORM,
    Texture.PixelFormat.RGBA_UINT_32: DXGI.DXGI_FORMAT_R32G32B32A32_UINT,
    Texture.PixelFormat.RG_UINT_32: DXGI.DXGI_FORMAT_R32G32_UINT,
    Texture.PixelFormat.R_UINT_32: DXGI.DXGI_FORMAT_R32_UINT,
    Texture.PixelFormat.RGBA_UINT_16: DXGI.DXGI_FORMAT_R16G16B16A16_UINT,
    Texture.PixelFormat.RG_UINT_16: DXGI.DXGI_FORMAT_R16G16_UINT,
    Texture.PixelFormat.R_UINT_16: DXGI.DXGI_FORMAT_R16_UINT,
    Texture.PixelFormat.RGBA_UINT_8: DXGI.DXGI_FORMAT_R8G8B8A8_UINT,
    Texture.PixelFormat.RG_UINT_8: DXGI.DXGI_FORMAT_R8G8_UINT,
    Texture.PixelFormat.R_UINT_8: DXGI.DXGI_FORMAT_R8_UINT,
    Texture.PixelFormat.RGBA_INT_32: DXGI.DXGI_FORMAT_R32G32B32A32_SINT,
    Texture.PixelFormat.RG_INT_32: DXGI.DXGI_FORMAT_R32G32_SINT,
    Texture.PixelFormat.R_INT_32: DXGI.DXGI_FORMAT_R32_SINT,
    Texture.PixelFormat.RGBA_INT_16: DXGI.DXGI_FORMAT_R16G16B16A16_SINT,
    Texture.PixelFormat.RG_INT_16: DXGI.DXGI_FORMAT_R16G16_SINT,
    Texture.PixelFormat.R_INT_16: DXGI.DXGI_FORMAT_R16_SINT,
    Texture.PixelFormat.RGBA_INT_8: DXGI.DXGI_FORMAT_R8G8B8A8_SINT,
    Texture.PixelFormat.RG_INT_8: DXGI.DXGI_FORMAT_R8G8_SINT,
    Texture.PixelFormat.R_INT_8: DXGI.DXGI_FORMAT_R8_SINT,
    Texture.PixelFormat.RGB_FLOAT_11_11_10: DXGI.DXGI_FORMAT_R11G11B10_FLOAT,
    Texture.PixelFormat.RGBA_UNORM_10_10_10_2: DXGI.DXGI_FORMAT_R10G10B10A2_UNORM,
    # Texture.PixelFormat.RGB_UNORM_11_11_10: ,
    Texture.PixelFormat.DEPTH_FLOAT_32_STENCIL_8: DXGI.DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    Texture.PixelFormat.DEPTH_FLOAT_32_STENCIL_0: DXGI.DXGI_FORMAT_D32_FLOAT,
    Texture.PixelFormat.DEPTH_24_STENCIL_8: DXGI.DXGI_FORMAT_D24_UNORM_S8_UINT,
    Texture.PixelFormat.DEPTH_16_STENCIL_0: DXGI.DXGI_FORMAT_D16_UNORM,
    Texture.PixelFormat.BC1: DXGI.DXGI_FORMAT_BC1_UNORM,  # Old: S3TC1
    Texture.PixelFormat.BC2: DXGI.DXGI_FORMAT_BC2_UNORM,  # Old: S3TC3
    Texture.PixelFormat.BC3: DXGI.DXGI_FORMAT_BC3_UNORM,  # Old: S3TC5
    Texture.PixelFormat.BC4U: DXGI.DXGI_FORMAT_BC4_UNORM,
    Texture.PixelFormat.BC4S: DXGI.DXGI_FORMAT_BC4_SNORM,
    Texture.PixelFormat.BC5U: DXGI.DXGI_FORMAT_BC5_UNORM,
    Texture.PixelFormat.BC5S: DXGI.DXGI_FORMAT_BC5_SNORM,
    Texture.PixelFormat.BC6U: DXGI.DXGI_FORMAT_BC6H_UF16,
    Texture.PixelFormat.BC6S: DXGI.DXGI_FORMAT_BC6H_SF16,
    Texture.PixelFormat.BC7: DXGI.DXGI_FORMAT_BC7_UNORM
}

class Bone:
    def __init__(self,matrix, index = 0):
        self.index = index
        self.matrix = matrix
class BoneData:
    def __init__(self,f,matrixCount):
        self.matrixCount = matrixCount
        self.boneList = []

        r = ByteReader
        self.indexOffset = f.tell()
        self.matrixOffset = self.indexOffset+(self.matrixCount*2)
        for i in range(self.matrixCount):
            f.seek(self.indexOffset)
            f.seek(i*2,1)
            index = r.int16(f)
            f.seek(self.matrixOffset+4)
            f.seek(64*i,1)
            matrix = Parse4x4Matrix(f)
            self.boneList.append(Bone(matrix,index))
            BoneMatrices[index] = matrix
class SkinnedMeshBoneBindings(DataBlock):
    def __init__(self, f,expectedGuid):
        super().__init__(f,BlockIDs["SkinnedMeshBoneBindings"],expectedGuid)
        r = ByteReader
        self.boneNameCount = r.uint32(f)
        self.boneNames = []
        for i in range(self.boneNameCount):
            self.boneNames.append(r.hashtext(f))
        self.matrixCount = r.uint32(f)
        self.boneData = BoneData(f,self.matrixCount)

        self.EndBlock(f)
class SkinnedMeshBoneBoundingBoxes(DataBlock):
    def __init__(self, f,expectedGuid):
        super().__init__(f,BlockIDs["SkinnedMeshBoneBoundingBoxes"],expectedGuid)
        r = ByteReader
        self.boneBoundingBoxesCount = r.uint32(f)
        self.boneBoundingBoxes = []
        for i in range(self.boneBoundingBoxesCount):
            self.boneBoundingBoxes.append(BoundingBox(f))
        self.indicesCount = r.uint32(f)
        self.indices = []
        for i in range(self.indicesCount):
            self.indices.append(r.uint16(f))
        self.usesIndices = r.bool(f)
        self.initialized = r.bool(f)
        self.EndBlock(f)

class DataBufferResource(DataBlock):
    def __init__(self, f,expectedGuid):
        super().__init__(f,BlockIDs["DataBufferResource"],expectedGuid)
        f.seek(20,1)
        r = ByteReader
        self.streamRef = r.path(f)
        self.dataOffset = r.uint64(f)
        self.datasize = r.uint64(f)
        self.EndBlock(f)
class MeshInfo:
    def __init__(self,f):
        r = ByteReader
        f.seek(4,1) #skip SkinInfoTypes
        f.seek(4,1) #skip SkinVtxTypes
        f.seek(16,1) #skip BlendShapeMask
        self.posVCount = f.tell()
        self.vertexCount = r.int32(f)
        f.seek(4,1) #skip VertexComputeNbtCount
        self.edgeRef = Reference(f)
        f.seek(4,1) #skip VerticesSkinCount
        f.seek(4,1) #skip VerticesSkinNbtCount
class RegularSkinnedMeshResourceSkinInfo(DataBlock): # was MeshBlockInfo
    def __init__(self,f,expectedGuid):
        super().__init__(f,BlockIDs["RegularSkinnedMeshResourceSkinInfo"],expectedGuid)
        r = ByteReader

        self.primitiveCount = r.uint32(f)
        self.meshInfos = []
        for i in range(self.primitiveCount):
            self.meshInfos.append(MeshInfo(f))
        #skip BlendTargetDeformation
        self.EndBlock(f)

        for m in self.meshInfos:
            if m.edgeRef.type.name == "Internal":
                self.edgeData = DataBufferResource(f,m.edgeRef.guid)


class StreamHandle:
    def __init__(self,f,streamStartOffset=0):
        r = ByteReader
        self.path = r.path(f)
        self.offsetPos = f.tell()
        self.resourceOffset = r.uint64(f)
        self.lengthPos = f.tell()
        self.resourceLength = r.uint64(f)
        self.absOffset = streamStartOffset + self.resourceOffset

class StreamData:
    class VertexElementDesc:
        class StorageType(IntEnum):
            Undefined = 0,
            SignedShortNormalized = 1,
            Float = 2,
            HalfFloat = 3,
            UnsignedByteNormalized = 4,
            SignedShort = 5,
            X10Y10Z10W2Normalized = 6,
            UnsignedByte = 7,
            UnsignedShort = 8,
            UnsignedShortNormalized = 9,
            UNorm8sRGB = 10,
            X10Y10Z10W2UNorm = 11
        class ElementType(IntEnum):
            Pos = 0,
            TangentBFlip = 1,
            Tangent = 2,
            Binormal = 3,
            Normal = 4,
            Color = 5,
            UV0 = 6,
            UV1 = 7,
            UV2 = 8,
            UV3 = 9,
            UV4 = 10,
            UV5 = 11,
            UV6 = 12,
            MotionVec = 13,
            Vec4Byte0 = 14,
            Vec4Byte1 = 15,
            BlendWeights = 16,
            BlendIndices = 17,
            BlendWeights2 = 18,
            BlendIndices2 = 19,
            PivotPoint = 20,
            AltPos = 21,
            AltTangent = 22,
            AltBinormal = 23,
            AltNormal = 24,
            AltColor = 25,
            AltUV0 = 26,
            Invalid = 27
        def __init__(self,f):
            r = ByteReader
            self.offset = r.uint8(f)
            storageTypeRead = r.uint8(f)
            typeValues = [item.value for item in self.StorageType]
            if storageTypeRead not in typeValues:
                raise Exception("Storage Type value at offset %d is invalid. Got %d" % (f.tell() - 1, storageTypeRead))
            else:
                self.storageType : StreamData.VertexElementDesc.StorageType = self.StorageType(storageTypeRead)
            self.count = r.uint8(f)
            elementTypeRead = r.uint8(f)
            typeValues = [item.value for item in self.ElementType]
            if elementTypeRead not in typeValues:
                raise Exception("Storage Type value at offset %d is invalid. Got %d" % (f.tell() - 1, elementTypeRead))
            else:
                self.elementType : StreamData.VertexElementDesc.ElementType = self.ElementType(elementTypeRead)
        def __str__(self):
            return "Offset: %d - StorageType %s - Count %d - ElementType %s"%(self.offset,self.storageType.name,self.count,self.elementType.name)

    def __init__(self,f,inStream,elementCount,streamStartOffset):
        r = ByteReader
        f.seek(4,1) #skip Flags
        self.stride = r.int32(f)
        self.elementInfoCount = r.int32(f)
        self.elementInfo = []
        for i in range(self.elementInfoCount):
            self.elementInfo.append(self.VertexElementDesc(f))
        f.seek(16,1) #skip GUID
        if inStream:
            self.streamInfo = StreamHandle(f,streamStartOffset)
            self.streamLength = self.streamInfo.resourceLength
            self.streamAbsOffset = self.streamInfo.absOffset
        else:
            self.streamLength = self.stride * elementCount
            self.streamAbsOffset = f.tell()
            f.seek(self.streamLength,1)

    def __str__(self):
        s = "\n'''''StreamData - Stride: %d  |  StreamLength: %d  |  StreamAbsOffset: %d"%(self.stride,self.streamLength,self.streamAbsOffset)
        for e in self.elementInfo:
            s += "\n''''''Element : " + e.__str__()
        return s
class Reference:
    class ReferenceType(IntEnum):
        Null = 0,
        Internal = 1,
        External = 2,
        Streaming = 3,
        UUID = 4
    def __init__(self,f):
        r = ByteReader
        typeRead = r.uint8(f)
        typeValues = [item.value for item in Reference.ReferenceType]
        if typeRead not in typeValues:
            raise Exception("Reference Type value at offset %d is invalid. Got %d"%(f.tell()-1,typeRead))
        else:
            self.type : Reference.ReferenceType = self.ReferenceType(typeRead)
            if self.type != 0:
                self.guid = r.guid(f)
                if self.type.name == "External":
                    self.externalFile = r.hashtext(f)

class VertexArrayResource(DataBlock):
    def __init__(self,f,expectedGuid):
        super().__init__(f,BlockIDs["VertexArrayResource"],expectedGuid)
        r = ByteReader
        self.posVCount = f.tell()
        self.vertexCount = r.uint32(f)
        self.streamRefCount = r.uint32(f)
        self.inStream = r.bool(f)
        streamStartOffset = 0
        self.vertexStream = StreamData(f,self.inStream,self.vertexCount,streamStartOffset)
        streamStartOffset += self.vertexStream.streamLength
        if self.streamRefCount == 3:
            self.normalsStream = StreamData(f,self.inStream,self.vertexCount,streamStartOffset)
            streamStartOffset += self.normalsStream.streamLength
            self.uvStream = StreamData(f, self.inStream, self.vertexCount,streamStartOffset)
        else:
            self.normalsStream = None
            self.uvStream = StreamData(f, self.inStream, self.vertexCount, streamStartOffset)

        self.EndBlock(f)
    def __str__(self):
        s = "\n--VertexArrayResource"
        s += "\n'''Vertex Count = %d  |  inStream = %s"%(self.vertexCount,str(self.inStream))
        s += "\n----VertexStream " + self.vertexStream.__str__()
        if self.normalsStream:
            s += "\n----NormalsStream " + self.normalsStream.__str__()
        s += "\n----UVStream " + self.uvStream.__str__()
        return s
class IndexArrayResource(DataBlock):
    def __init__(self,f,expectedGuid):
        super().__init__(f,BlockIDs["IndexArrayResource"],expectedGuid)
        r = ByteReader
        self.posIndexCount = f.tell()
        self.indexCount = r.int32(f)
        f.seek(4,1) #skip Flags
        self.indexFormat = r.uint32(f) #0 = 16bit, 1 = 32bit
        self.inStream = r.bool(f)
        f.seek(3,1)
        f.seek(16,1) #skip GUID
        if self.inStream:
            self.indexStream = StreamHandle(f)
        else:
            if self.indexFormat == 0:
                formatStride = 16
            else:
                formatStride = 32
            f.seek(formatStride * self.indexCount,1) #skip
        self.EndBlock(f)

class TextureRef():
    def __init__(self,f):
        self.texPath = ""

        r = ByteReader
        f.seek(16,1)
        indicator = r.int8(f)
        f.seek(16,1)
        if indicator == 2:
            self.texPath = r.hashtext(f)
        f.seek(16,1)

class RenderTechnique:
    class RenderTechniqueState:
        def __init__(self,f):
            f.seek(8,1) #skip PackedData/DepthBias/ColorMask
    class SRTBindingCache:
        def __init__(self,f):
            r = ByteReader
            f.seek(1,1) #skip TextureBindingMask
            f.seek(2,1) #skip BindingDataMask
            f.seek(8,1) #skip SRTEntriesMask
            indicesCount = r.uint32(f)
            f.seek(indicesCount*2,1) #skip BindingDataIndices
            handlesCount = r.uint32(f)
            f.seek(handlesCount*8,1) #skip HwBindingHandle
    class ShaderSamplerBinding:
        def __init__(self,f):
            f.seek(4,1) #skip BindingNameHash
            f.seek(4,1) #skip SamplerData
            f.seek(8,1) #skip SamplerBindingHandle
    class ShaderTextureBinding:
        def __init__(self,f):
            r = ByteReader
            f.seek(16,1) #skip BindingNameHash, BindingSwizzleNameHash, SamplerNameHash, PackedData uint32s
            self.textureResource = Reference(f)
            f.seek(16,1) #skip BindingHandles
    class ShaderVariableBinding:
        def __init__(self,f):
            r = ByteReader
            f.seek(8,1) #skip BindingNameHash, VariableIDHash
            f.seek(1,1) #skip VariableType
            f.seek(16,1) #skip VariableData
            f.seek(8,1) #skip VarBindingHandle

    def __init__(self,f):
        r = ByteReader
        self.RenderTechniqueState(f)
        self.SRTBindingCache(f)
        f.seek(4,1) #skip Technique Type
        f.seek(8,1) #skip WorldDataBindingMask
        f.seek(3,1) #skip bools GPUSkinned, WriteGlobalVertexCache, InitiallyEnabled
        f.seek(4,1) #skip MaterialLayerID
        samplerCount = r.uint32(f)
        for i in range(samplerCount):
            self.ShaderSamplerBinding(f)
        self.texCount = r.uint32(f)
        self.shaderTextureBindings = []
        self.textureBlockCount = 0
        for i in range(self.texCount):
            stb = self.ShaderTextureBinding(f)
            if stb.textureResource.type.name == "Internal":
                self.textureBlockCount += 1  # we use this to find color ramp textures among the shader blocks
            self.shaderTextureBindings.append(stb)
        self.varCount = r.uint32(f)
        for i in range(self.varCount):
            self.ShaderVariableBinding(f)
        self.shaderRef = Reference(f)
        f.seek(8,1) #skip ID
class RenderTechniqueSet:
    def __init__(self,f):
        r = ByteReader
        self.techniquesCount = r.uint32(f)
        self.renderTechniques = []
        for i in range(self.techniquesCount):
            self.renderTechniques.append(RenderTechnique(f))
        f.seek(4,1) #skip TechniqueSetType
        f.seek(4,1) #skip RenderEffectType
        f.seek(8,1) #skip TechniquesMask variables
class ShaderResource(DataBlock):
    def __init__(self,f,expectedGuid):
        super().__init__(f,BlockIDs["ShaderResource"],expectedGuid)
        #skip everything
        self.EndBlock(f)
class RenderEffectResource(DataBlock):
    def __init__(self,f,expectedGuid):
        super().__init__(f,BlockIDs["RenderEffectResource"],expectedGuid)
        self.shaderName = ""
        self.ui_ShowTextures = False
        self.uniqueTextures = []

        r = ByteReader
        self.shaderName = r.hashtext(f)
        self.techniqueSetsCount = r.uint32(f)
        self.techniqueSets = []
        for i in range(self.techniqueSetsCount):
            self.techniqueSets.append(RenderTechniqueSet(f))
        f.seek(8,1) #skip SortMode and SortOrder
        f.seek(4,1) #skip EffectType
        f.seek(1,1) #skip MakeAccumulationBufferCopy
        f.seek(4,1) #skip VertexElementSet

        self.EndBlock(f)
        textureGUIDs = []
        for ts in self.techniqueSets:
            for tr in ts.renderTechniques:
                ShaderResource(f,tr.shaderRef.guid)
                for tbc in tr.shaderTextureBindings:
                    if tbc.textureResource.type.name == 'Internal':
                        if tbc.textureResource.guid not in textureGUIDs:
                            t = DataBlock(f, BlockIDs["TextureResource"],tbc.textureResource.guid)  # TODO skipping internal texture for now
                            textureGUIDs.append(t.guid)
                            t.EndBlock(f)

        self.GetUniqueTexturesOfMatIndex()

    def GetUniqueTexturesOfMatIndex(self):
        for ts in self.techniqueSets:
            for tr in ts.renderTechniques:
                for tb in tr.shaderTextureBindings:
                    if tb.textureResource.type.name == "External":
                        if tb.textureResource.externalFile not in self.uniqueTextures:
                            self.uniqueTextures.append(tb.textureResource.externalFile)
        return self.uniqueTextures

class RenderingPrimitiveResource(DataBlock):
    def __init__(self, f,expectedGuid):
        super().__init__(f, BlockIDs["RenderingPrimitiveResource"],expectedGuid)
        r = ByteReader

        f.seek(4,1) #skip Flags
        self.vertexRef = Reference(f)
        self.indexRef = Reference(f)
        self.boxExtents = BoundingBox(f)
        self.indexOffset = r.int32(f)
        self.skdTreeRef = Reference(f)
        self.startIndex = r.int32(f)
        self.posIndexCount = f.tell()
        self.endIndex = r.int32(f) #Index Count ( tris count / 3)
        f.seek(4,1) #skip hash
        self.renderEffectRef = Reference(f)

        self.EndBlock(f)

        if self.vertexRef.type != 0:
            self.vertexBlock = VertexArrayResource(f,self.vertexRef.guid)
        if self.indexRef.type != 0:
            self.faceBlock = IndexArrayResource(f,self.indexRef.guid)
        if self.skdTreeRef.type != 0:
            self.skdTreeResource = DataBlock(f,BlockIDs["SKDTreeResource"],self.skdTreeRef.guid) #ignore SKDT
            self.skdTreeResource.EndBlock(f)
        if self.renderEffectRef.type != 0:
            self.renderEffectResource = RenderEffectResource(f,self.renderEffectRef.guid)
    def __str__(self):

        return "PRIMITIVE" + self.vertexBlock.__str__()

class StaticMeshResource(DataBlock):
    def __init__(self,f,expectedGuid=0):
        super().__init__(f,BlockIDs["StaticMeshResource"],expectedGuid)
        r = ByteReader

        self.meshName = r.hashtext(f)
        self.meshBase = MeshResourceBase(f)
        f.seek(4,1) #skip DrawFlags
        self.primitiveCount = r.uint32(f)
        self.primitiveRefs = []
        for i in range(self.primitiveCount):
            self.primitiveRefs.append(Reference(f))
        self.materialCount = r.int32(f)
        self.materialRefs = []
        for i in range(self.materialCount):
            self.materialRefs.append(Reference(f))
        self.SkeletonHelpersRef = Reference(f)
        #skip SimulationInfoRef
        #skip Bool SupportsInstanceRendering
        self.EndBlock(f)

        self.primitives = []
        for pf in self.primitiveRefs:
            self.primitives.append(RenderingPrimitiveResource(f, pf.guid))

class RegularSkinnedMeshResource(DataBlock):
    def __init__(self, f,expectedGuid=0):
        super().__init__(f,BlockIDs["RegularSkinnedMeshResource"],expectedGuid)
        r = ByteReader

        self.meshName = r.hashtext(f)
        self.meshBase = MeshResourceBase(f)
        self.skeletonRef = Reference(f)
        self.SkeletonHelpersRef = Reference(f)
        f.seek(8,1) #skip DrawFlags
        self.boneBindingsRef = Reference(f)
        self.boneBoundingBoxesRef = Reference(f)
        self.positionBoundsScale = r.vector3(f)
        self.positionBoundsOffset = r.vector3(f)
        self.skinInfoRef = Reference(f)
        self.primitiveCount = r.uint32(f)
        self.primitiveRefs = []
        for i in range(self.primitiveCount):
            self.primitiveRefs.append(Reference(f))
        self.materialCount = r.int32(f)
        self.materialRefs = []
        for i in range(self.materialCount):
            self.materialRefs.append(Reference(f))

        self.EndBlock(f)

        self.boneBindings = SkinnedMeshBoneBindings(f,self.boneBindingsRef.guid)
        self.boneBoundingBoxes = SkinnedMeshBoneBoundingBoxes(f,self.boneBoundingBoxesRef.guid)
        self.skinInfo = RegularSkinnedMeshResourceSkinInfo(f,self.skinInfoRef.guid)

        self.primitives = []
        for pf in self.primitiveRefs:
            self.primitives.append(RenderingPrimitiveResource(f,pf.guid))

        self.materials = []
        for mr in self.materialRefs:
            self.materials.append(RenderEffectResource(f,mr.guid))

class BoundingBox:
    def __init__(self,f):
        r = ByteReader
        self.minExtent = r.vector3(f)
        self.maxExtent = r.vector3(f)
class Transform:
    class RotationMatrix:
        def __init__(self,f):
            r = ByteReader
            self.Col1 = r.vector3(f)
            self.Col2 = r.vector3(f)
            self.Col3 = r.vector3(f)
    class WorldPosition:
        def __init__(self,f):
            r = ByteReader
            self.pos = r.dvector3(f)
    def __init__(self,f):
        self.rotationMatrix = self.RotationMatrix(f)
        self.worldPosition = self.WorldPosition(f)
class CullInfo:
    class MeshType(IntEnum):
        RegularSkinnedMesh = 32,
        StaticMesh = -32
    def __init__(self,f):
        r = ByteReader
        typeRead = r.int8(f)
        if typeRead == -32 or typeRead == 32:
            self.meshType = self.MeshType(typeRead)
        elif typeRead == 0:
            self.meshType = self.MeshType(32)
            pass #it's fine I guess
        else:
            raise Exception("Mesh Type is invalid, expected -32 or 32, got %d at offset %d"%(typeRead,f.tell()-1))
        f.seek(3,1)
class MeshHierarchyInfo:
    def __init__(self,f):
        r = ByteReader
        self.MITNodeSize = r.uint32(f)
        self.primitiveCount = r.uint32(f)
        self.meshCount = r.uint16(f)
        self.staticMeshCount = r.uint16(f)
        self.lodMeshCount = r.uint16(f)
        self.packedData = r.uint16(f)
class MeshResourceBase:
    def __init__(self,f):
        self.boundingBox = BoundingBox(f)
        self.drawableCullInfo = CullInfo(f)
        self.meshHierarchyInfo = MeshHierarchyInfo(f)
        f.seek(4,1) #skip StaticDataBlockSize


class LodMeshResource(DataBlock): # was LODGroup
    def __init__(self, f):
        super().__init__(f)
        self.objectName = ""
        self.totalMeshCount = 0
        self.LODList = []


        r = ByteReader
        HZDEditor = bpy.context.scene.HZDEditor
        self.objectName = r.hashtext(f)
        self.meshBase = MeshResourceBase(f)
        f.seek(4,1) #skip MaxDistance
        self.meshCount = r.int32(f)
        self.meshRefs = []
        self.LODDistanceList = []
        for i in range(self.meshCount):
            self.meshRefs.append(Reference(f))
            lodDistance = r.float(f)
            HZDEditor["LodDistance" + str(i)] = lodDistance
            self.LODDistanceList.append(lodDistance)

        self.EndBlock(f)

        self.meshList = []
        if len(asset.MultiMeshResources) != 0:
            multiMeshGuid = asset.MultiMeshResources[0].guid
        else:
            multiMeshGuid = 0
        for rp in self.meshRefs:
            if rp.guid != multiMeshGuid:
                if self.meshBase.drawableCullInfo.meshType == 32:
                    self.meshList.append(RegularSkinnedMeshResource(f, expectedGuid=rp.guid))
                elif self.meshBase.drawableCullInfo.meshType == -32:
                    self.meshList.append(StaticMeshResource(f, expectedGuid=rp.guid))
class MultiMeshResource(DataBlock): # was LODObject
    class ResourcePart:
        def __init__(self,f):
            self.meshRef = Reference(f)
            self.transform = Transform(f)
    def __init__(self, f):
        super().__init__(f,BlockIDs["MultiMeshResource"])
        r = ByteReader

        self.objectName = r.hashtext(f)
        self.meshBase = MeshResourceBase(f)
        self.meshCount = r.int32(f)
        self.meshRefs = []
        for i in range(self.meshCount):
            self.meshRefs.append(self.ResourcePart(f))

        self.EndBlock(f)

        self.meshList = []
        for rp in self.meshRefs:
            if self.meshBase.drawableCullInfo.meshType == 32:
                self.meshList.append(RegularSkinnedMeshResource(f,expectedGuid=rp.meshRef.guid))
            elif self.meshBase.drawableCullInfo.meshType == -32:
                self.meshList.append(StaticMeshResource(f,expectedGuid=rp.meshRef.guid))

def ImportAllMeshes(maxLod = 99):
    for mmi, lm in enumerate(asset.MultiMeshResources):
        for mi, m in enumerate(lm.meshList):
            if mi <= maxLod:
                for pi, p in enumerate(m.primitives):
                    ImportMesh(False, mmi, mi, pi)
    for lmi, lm in enumerate(asset.LodMeshResources):
        for mi, m in enumerate(lm.meshList):
            if mi <= maxLod:
                for pi, p in enumerate(m.primitives):
                    ImportMesh(True, lmi, mi, pi)
def ReadCoreFile():
    r = ByteReader
    HZDEditor = bpy.context.scene.HZDEditor
    global asset
    asset = Asset()

    core = Path(HZDEditor.HZDAbsPath)
    coresize = os.path.getsize(core)
    with open(core, "rb") as f:
        while f.tell() < coresize:
            # print(f.tell(),coresize)
            ID = r.uint64(f)
            f.seek(-8,1)
            # print(ID)
            if ID == BlockIDs["LodMeshResource"]: # LodMeshResource
                asset.LodMeshResources.append(LodMeshResource(f))
            elif ID == BlockIDs["MultiMeshResource"]: # MultiMeshResource
                asset.MultiMeshResources.append(MultiMeshResource(f))
            elif ID == BlockIDs["RegularSkinnedMeshResource"]: # RegularSkinnedMeshResource
                asset.RegularSkinnedMeshResources.append(RegularSkinnedMeshResource(f))
            elif ID == BlockIDs["StaticMeshResource"]: # StaticMeshResource
                asset.StaticMeshResources.append(StaticMeshResource(f))
            else:
                raise Exception("This file is not supported.",ID)
            #TODO need support for RegularSkinnedMeshResources that are not part of a Lod or MultiMesh
            #W:\HorizonModding\Extract\models\characters\robots\horse\animation\parts\firesheets.core

class SearchForOffsets(bpy.types.Operator):
    """Searches the .core file for offsets and sizes"""
    bl_idname = "object.hzd_offsets"
    bl_label = "Search Data"

    def execute(self,context):
        HZDEditor = bpy.context.scene.HZDEditor

        # TODO this shouldn't be in here
        HZDEditor.HZDPath = HZDEditor.HZDPath.strip('\"')
        HZDEditor.GamePath = HZDEditor.GamePath.strip('\"')
        HZDEditor.WorkPath = HZDEditor.WorkPath.strip('\"')
        HZDEditor.SkeletonPath = HZDEditor.SkeletonPath.strip('\"')
        core = Path(HZDEditor.HZDAbsPath)

        if core.is_dir():
            print("Path is a folder, attempting to import every mesh found in: \n", core)
            coreFiles = []
            glob = core.glob('**/*.core')
            if glob:
                print("FILES FOUND: ")
                for c in glob:
                    print(c)
                    coreFiles.append(c)
                print("\n\n")
            else:
                raise Exception("No .core file found in: ", core)

            if coreFiles:
                print("IMPORTING FILES: ")
                for c in coreFiles:
                    HZDEditor.HZDPath = str(c)
                    ReadCoreFile()
                    ImportAllMeshes(0)

        else:
            ReadCoreFile()
        return{'FINISHED'}
class ExtractHZDAsset(bpy.types.Operator):
    """Extract Asset directly from Horizon .bin files"""
    bl_idname = "object.extract_asset"
    bl_label = "Extract"

    def execute(self,context):
        HZDEditor = context.scene.HZDEditor
        ExtractAsset(HZDEditor.AssetPath)
        return {'FINISHED'}

class ImportAll(bpy.types.Operator):
    """Imports absolutely every meshes."""
    bl_idname = "object.import_all"
    bl_label = "Import All"

    def execute(self,context):
        ImportAllMeshes()
        return {'FINISHED'}

class ImportHZD(bpy.types.Operator):
    """Imports the mesh"""
    bl_idname = "object.import_hzd"
    bl_label = ""

    isSingular: bpy.props.BoolProperty() # if singular mesh there is no LodMesh or MultiMesh
    isLodMesh: bpy.props.BoolProperty() # is LodMeshResource, else it's a MultiMeshResource
    resourceIndex: bpy.props.IntProperty() # within the resource array, which is it (usually 0)
    meshIndex: bpy.props.IntProperty() # within the meshList of Resource, which is it (this points to either RegularSkinnedMeshResource or StaticMeshResource)
    primitiveIndex: bpy.props.IntProperty() # within the Static or Skinned mesh resource, which Primitive is it?


    def execute(self, context):
        ImportMesh(self.isLodMesh,self.resourceIndex,self.meshIndex,self.primitiveIndex)
        return {'FINISHED'}
class ImportLodHZD(bpy.types.Operator):
    """Imports every mesh in the LOD"""
    bl_idname = "object.import_lod_hzd"
    bl_label = "Import"

    isLodMesh: bpy.props.BoolProperty()  # is LodMeshResource, else it's a MultiMeshResource
    resourceIndex: bpy.props.IntProperty()  # within the resource array, which is it (usually 0)
    meshIndex: bpy.props.IntProperty()  # within the meshList of Resource, which is it (this points to either RegularSkinnedMeshResource or StaticMeshResource)
    # dont care about primitive index, we want them all
    def execute(self, context):
        if self.isLodMesh:
            for primitiveIndex,primitive in enumerate(asset.LodMeshResources[self.resourceIndex].meshList[self.meshIndex].primitives):
                ImportMesh(self.isLodMesh,self.resourceIndex,self.meshIndex,primitiveIndex)
        else:
            for primitiveIndex, primitive in enumerate(asset.MultiMeshResources[self.resourceIndex].meshList[self.meshIndex].primitives):
                ImportMesh(self.isLodMesh, self.resourceIndex, self.meshIndex, primitiveIndex)
        return {'FINISHED'}
class ImportSkeleton(bpy.types.Operator):
    """Creates a skeleton"""
    bl_idname = "object.import_skt"
    bl_label = "Import Skeleton"

    def execute(self, context):
        o = CreateSkeleton()
        return {'FINISHED'}
class ExportHZD(bpy.types.Operator):
    """Exports the mesh based on object name"""
    bl_idname = "object.export_hzd"
    bl_label = ""

    isLodMesh: bpy.props.BoolProperty()
    resourceIndex: bpy.props.IntProperty()
    meshIndex: bpy.props.IntProperty()
    primitiveIndex: bpy.props.IntProperty()

    def execute(self, context):
        ExportMesh(self.isLodMesh,self.resourceIndex,self.meshIndex,self.primitiveIndex)
        return {'FINISHED'}
class ExportLodHZD(bpy.types.Operator):
    """Exports every mesh in the LOD"""
    bl_idname = "object.export_lod_hzd"
    bl_label = "Export"

    isLodMesh: bpy.props.BoolProperty()
    resourceIndex: bpy.props.IntProperty()
    meshIndex: bpy.props.IntProperty()


    def execute(self, context):
        if self.isLodMesh:
            for primitiveIndex,primitive in enumerate(asset.LodMeshResources[self.resourceIndex].meshList[self.meshIndex].primitives):
                ExportMesh(self.isLodMesh,self.resourceIndex,self.meshIndex,primitiveIndex)
                ReadCoreFile()

        else:
            for primitiveIndex,primitive in enumerate(asset.MultiMeshResources[self.resourceIndex].meshList[self.meshIndex].primitives):
                ExportMesh(self.isLodMesh,self.resourceIndex,self.meshIndex,primitiveIndex)
                ReadCoreFile()

        return {'FINISHED'}
class SaveLodDistances(bpy.types.Operator):
    """Save all LOD Distances"""
    bl_idname = "object.savedistances"
    bl_label = "Save LOD Distances"

    Index: bpy.props.IntProperty()

    def execute(self, context):
        SaveDistances( self.Index)
        return {'FINISHED'}
class ShowUsedTextures(bpy.types.Operator):
    """Show used texture paths for the mesh block"""
    bl_idname = "object.usedtextures"
    bl_label = ""

    isLodMesh: bpy.props.BoolProperty()  # is LodMeshResource, else it's a MultiMeshResource
    resourceIndex: bpy.props.IntProperty()  # within the resource array, which is it (usually 0)
    meshIndex: bpy.props.IntProperty()  # within the meshList of Resource, which is it (this points to either RegularSkinnedMeshResource or StaticMeshResource)
    primitiveIndex: bpy.props.IntProperty()  # within the Static or Skinned mesh resource, which Primitive is it?
    def execute(self,context):
        if self.isLodMesh:
            p = asset.LodMeshResources[self.resourceIndex].meshList[self.meshIndex].primitives[self.primitiveIndex]
            if p.renderEffectRef.type != 0:  # Primitive has a material
                mat = p.renderEffectResource
            else:
                mat = asset.LodMeshResources[self.resourceIndex].meshList[self.meshIndex].materials[self.primitiveIndex]
            mat.ui_ShowTextures = not mat.ui_ShowTextures
        else:
            p = asset.MultiMeshResources[self.resourceIndex].meshList[self.meshIndex].primitives[self.primitiveIndex]
            if p.renderEffectRef.type != 0:  # Primitive has a material
                mat = p.renderEffectResource
            else:
                mat = asset.MultiMeshResources[self.resourceIndex].meshList[self.meshIndex].materials[self.primitiveIndex]
            mat.ui_ShowTextures = not mat.ui_ShowTextures

        return {'FINISHED'}

class HZDPanel(bpy.types.Panel):
    """Creates a Panel in the Scene Properties window"""
    bl_label = "Horizon Zero Dawn"
    bl_idname = "OBJECT_PT_hzdpanel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self,context):
        layout = self.layout
        HZDEditor = context.scene.HZDEditor

        row = layout.row()
        row.prop(HZDEditor, "AssetPath")
        row.operator("object.extract_asset",icon="EXPORT")

        row = layout.row()

        row = layout.row()
        row.prop(HZDEditor, "WorkPath")
        row = layout.row()
        row.prop(HZDEditor, "GamePath")

        row = layout.row()
        row.prop(HZDEditor,"HZDPath")
        row = layout.row()
        row.prop(HZDEditor, "SkeletonPath")

        row = layout.row()
        row.prop(HZDEditor, "NVTTPath")

        row = layout.row()
        row.operator("object.hzd_offsets", icon='ZOOM_ALL')
        if Asset:
            row = layout.row()
            row.operator("object.import_skt",icon="ARMATURE_DATA")
            row = layout.row()
            row.prop(HZDEditor,"ExtractTextures")
            row.prop(HZDEditor,"OverwriteTextures")
            row = layout.row()
            row.operator("object.import_all",icon="OUTLINER_OB_LIGHTPROBE")
        mainRow = layout.row()
class LODDistancePanel(bpy.types.Panel):
    bl_label = "LOD Distances"
    bl_idname = "OBJECT_PT_loddist"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    bl_parent_id = "OBJECT_PT_hzdpanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        if asset:
            layout = self.layout
            HZDEditor = context.scene.HZDEditor
            mainRow = layout.row()
            for ig, lg in enumerate(asset.LodMeshResources):
                box = mainRow.box()
                box.label(text="LOD DISTANCES", icon='OPTIONS')
                saveDistances = box.operator("object.savedistances")
                saveDistances.Index = ig
                for il, l in enumerate(lg.meshList):
                    lodBox = box.box()
                    disRow = lodBox.row()
                    disRow.prop(HZDEditor, "LodDistance" + str(il))
class MultiMeshPanel(bpy.types.Panel):
    bl_label = "Multi Mesh"
    bl_idname = "OBJECT_PT_multimesh"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    bl_parent_id = "OBJECT_PT_hzdpanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        if asset:
            layout = self.layout
            HZDEditor = context.scene.HZDEditor
            for imm, mm in enumerate(asset.MultiMeshResources):
                mainRow = layout.row()
                box = mainRow.box()
                box.label(text="MULTI MESH", icon='SNAP_VOLUME')
                for il, l in enumerate(mm.meshList):
                    lodBox = box.box()
                    lodRow = lodBox.row()
                    lodRow.label(text="PART", icon='MATERIAL_DATA')
                    #Big buttons that import whole LODs/Parts
                    LODImport = lodRow.operator("object.import_lod_hzd", icon='IMPORT')
                    LODImport.isLodMesh = False
                    LODImport.resourceIndex = imm
                    LODImport.meshIndex = il
                    LODExport = lodRow.operator("object.export_lod_hzd", icon='EXPORT')
                    LODExport.isLodMesh = False
                    LODExport.resourceIndex = imm
                    LODExport.meshIndex = il
                    for ip, p in enumerate(l.primitives):
                        row = lodBox.row()
                        row.label(text=str(ip) + "_" + l.meshName + " " + str(p.vertexBlock.vertexCount),
                                  icon='MESH_ICOSPHERE')
                        if p.vertexBlock.inStream:
                            if p.renderEffectRef.type != 0: #Primitive has a material
                                mat = p.renderEffectResource
                            else:
                                mat = l.materials[ip]
                            if mat.ui_ShowTextures:
                                texIcon = 'UV'
                            else:
                                texIcon = 'TEXTURE'

                            texButton = row.operator("object.usedtextures", icon=texIcon)
                            texButton.isLodMesh = False
                            texButton.resourceIndex = imm
                            texButton.meshIndex = il
                            texButton.primitiveIndex = ip

                            Button = row.operator("object.import_hzd", icon='IMPORT')
                            Button.isLodMesh = False
                            Button.resourceIndex = imm
                            Button.meshIndex = il
                            Button.primitiveIndex = ip
                            Button = row.operator("object.export_hzd", icon='EXPORT')
                            Button.isLodMesh = False
                            Button.resourceIndex = imm
                            Button.meshIndex = il
                            Button.primitiveIndex = ip

                            if mat.ui_ShowTextures:
                                texBox = lodBox.box()

                                for t in mat.uniqueTextures:
                                    texRow = texBox.row()
                                    texRow.label(text=t)

                        else:
                            row.label(text="Not able to Import for now.")
class LodGroupPanel(bpy.types.Panel):
    bl_label = "LOD Meshes"
    bl_idname = "OBJECT_PT_lodgroup"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    bl_parent_id = "OBJECT_PT_hzdpanel"
    bl_options = {"DEFAULT_CLOSED"}
    def draw(self, context):
        if asset:
            layout = self.layout
            HZDEditor = context.scene.HZDEditor
            mainRow = layout.row()
            for ilm, lm in enumerate(asset.LodMeshResources):
                box = mainRow.box()
                box.label(text="LOD MESH", icon='STICKY_UVS_LOC')
                for il, l in enumerate(lm.meshList):
                    lodBox = box.box()
                    lodRow = lodBox.row()
                    lodRow.label(text="LOD", icon='MOD_EXPLODE')
                    #Big buttons that import whole LODs/Parts
                    LODImport = lodRow.operator("object.import_lod_hzd", icon='IMPORT')
                    LODImport.isLodMesh = True
                    LODImport.resourceIndex = ilm
                    LODImport.meshIndex = il
                    LODExport = lodRow.operator("object.export_lod_hzd", icon='EXPORT')
                    LODExport.isLodMesh = True
                    LODExport.resourceIndex = ilm
                    LODExport.meshIndex = il
                    for ip, p in enumerate(l.primitives):
                        row = lodBox.row()
                        row.label(text=str(ip) + "_" + l.meshName + " " + str(p.vertexBlock.vertexCount),
                                  icon='MESH_ICOSPHERE')
                        if p.vertexBlock.inStream:
                            if p.renderEffectRef.type != 0: #Primitive has a material
                                mat = p.renderEffectResource
                            else:
                                mat = l.materials[ip]
                            if mat.ui_ShowTextures:
                                texIcon = 'UV'
                            else:
                                texIcon = 'TEXTURE'

                            texButton = row.operator("object.usedtextures", icon=texIcon)
                            texButton.isLodMesh = True
                            texButton.resourceIndex = ilm
                            texButton.meshIndex = il
                            texButton.primitiveIndex = ip

                            Button = row.operator("object.import_hzd", icon='IMPORT')
                            Button.isLodMesh = True
                            Button.resourceIndex = ilm
                            Button.meshIndex = il
                            Button.primitiveIndex = ip
                            Button = row.operator("object.export_hzd", icon='EXPORT')
                            Button.isLodMesh = True
                            Button.resourceIndex = ilm
                            Button.meshIndex = il
                            Button.primitiveIndex = ip

                            if mat.ui_ShowTextures:
                                texBox = lodBox.box()

                                for t in mat.uniqueTextures:
                                    texRow = texBox.row()
                                    texRow.label(text=t)

                        else:
                            row.label(text="Not able to Import for now.")
class SingularMeshPanel(bpy.types.Panel):
    bl_label = "Singular Meshes"
    bl_idname = "OBJECT_PT_singularmesh"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    bl_parent_id = "OBJECT_PT_hzdpanel"
    bl_options = {"DEFAULT_CLOSED"}
    def draw(self, context):
        if asset:
            layout = self.layout
            HZDEditor = context.scene.HZDEditor
            mainRow = layout.row()
            for il, l in enumerate(asset.RegularSkinnedMeshResources):
                box = mainRow.box()
                box.label(text="SINGULAR MESH", icon='STICKY_UVS_LOC')

                lodBox = box.box()
                lodRow = lodBox.row()
                lodRow.label(text="NOT SUPPORTED", icon='MOD_EXPLODE')
                #Big buttons that import whole LODs/Parts
                # LODImport = lodRow.operator("object.import_lod_hzd", icon='IMPORT')
                # LODImport.isLodMesh = True
                # LODImport.resourceIndex = il
                # LODImport.meshIndex = -1 # in this case the resource is the mesh
                # LODExport = lodRow.operator("object.export_lod_hzd", icon='EXPORT')
                # LODExport.isLodMesh = True
                # LODExport.resourceIndex = il
                # LODExport.meshIndex = -1
                for ip, p in enumerate(l.primitives):
                    row = lodBox.row()
                    row.label(text=str(ip) + "_" + l.meshName + " " + str(p.vertexBlock.vertexCount),
                              icon='MESH_ICOSPHERE')
                    # if p.vertexBlock.inStream:
                    #     if p.renderEffectRef.type != 0: #Primitive has a material
                    #         mat = p.renderEffectResource
                    #     else:
                    #         mat = l.materials[ip]
                    #     if mat.ui_ShowTextures:
                    #         texIcon = 'UV'
                    #     else:
                    #         texIcon = 'TEXTURE'
                    #
                    #     texButton = row.operator("object.usedtextures", icon=texIcon)
                    #     texButton.isLodMesh = True
                    #     texButton.resourceIndex = il
                    #     texButton.meshIndex = -1
                    #     texButton.primitiveIndex = ip
                    #
                    #     Button = row.operator("object.import_hzd", icon='IMPORT')
                    #     Button.isLodMesh = True
                    #     Button.resourceIndex = il
                    #     Button.meshIndex = -1
                    #     Button.primitiveIndex = ip
                    #     Button = row.operator("object.export_hzd", icon='EXPORT')
                    #     Button.isLodMesh = True
                    #     Button.resourceIndex = il
                    #     Button.meshIndex = -1
                    #     Button.primitiveIndex = ip
                    #
                    #     if mat.ui_ShowTextures:
                    #         texBox = lodBox.box()
                    #
                    #         for t in mat.uniqueTextures:
                    #             texRow = texBox.row()
                    #             texRow.label(text=t)


                    row.label(text="Not able to Import for now.")

classes=[ImportHZD,
         ImportAll,
         ImportLodHZD,
         ImportSkeleton,
         ExportHZD,
         ExportLodHZD,
         SaveLodDistances,
         HZDSettings,
         SearchForOffsets,
         HZDPanel,
         LODDistancePanel,
         MultiMeshPanel,
         LodGroupPanel,
         ShowUsedTextures,
         ExtractHZDAsset,
         SingularMeshPanel,
         NODE_OT_HZDMT_add,
         NODE_MT_HZDMT_add]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.HZDEditor = bpy.props.PointerProperty(type=HZDSettings)
    bpy.types.NODE_MT_add.append(add_node_button)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    bpy.types.NODE_MT_add.remove(add_node_button)

if __name__ == "__main__":
    register()
