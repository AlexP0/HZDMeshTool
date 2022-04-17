bl_info = {
    "name": "HZD Mesh Tool",
    "author": "AlexPo",
    "location": "Scene Properties > HZD Panel",
    "version": (1, 3, 2),
    "blender": (3, 1, 0),
    "description": "This addon imports/exports skeletal meshes\n from Horizon Zero Dawn's .core/.stream files",
    "category": "Import-Export"
    }


# def install_Package():
#     import subprocess
#     import sys
#     import os
#
#     python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
#
#     subprocess.call([python_exe, "-m", "ensurepip"])
#     subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
#
#     subprocess.call([python_exe, "-m", "pip", "install", "mmh3"])
# install_Package()

import bpy
import bmesh
import os
import pathlib
from struct import unpack, pack
import numpy as np
import mathutils
import operator
# import mmh3
#from . \
import pymmh3

import ctypes
from ctypes import c_size_t, c_char_p, c_int32
from pathlib import Path
from typing import Union, Dict
from enum import IntEnum

if "bpy" in locals():
    import imp
    imp.reload(pymmh3)
else:
    from . import pymmh3


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

BoneMatrices = {}

verbose = True

def say(string):
    if verbose:
        print(str(string))

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
            self.filesize = r.int64(f)
            self.datasize = r.int64(f)
            self.filecount = r.int64(f)
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
            self.id = r.int32(f)
            self.key0 = r.int32(f)
            self.hash = r.int64(f)
            self.offset = r.int64(f)
            self.size = r.int32(f)
            self.key1 = r.int32(f)

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
            self.uncompressed_offset = r.int64(f)
            self.uncompressed_size = r.int32(f)
            self.key0 = r.int32(f)
            self.compressed_offset = r.int64(f)
            self.compressed_size = r.int32(f)
            self.key1 = r.int32(f)

        def write(self, f):
            w = BytePacker
            bChunkEntry = b''
            bChunkEntry += w.int64(self.uncompressed_offset)
            bChunkEntry += w.int32(self.uncompressed_size)
            bChunkEntry += w.int32(self.key0)
            bChunkEntry += w.int64(self.compressed_offset)
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
        # bHash = BytePacker.int64(fileHash)
        # print(bHash)
        # say("mmh3 = "+hex(fileHash)+" ("+string+")")
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

    def ExtractFile(self,file,filePath, isStream = False):
        if os.path.exists(bpy.context.scene.HZDEditor.GameAbsPath):
            say("Game Path is Valid")
            ## I do not know who the original author of this Oodle class is
            ## I got it from here https://github.com/REDxEYE/ProjectDecima_python/tree/master/ProjectDecima/utils
            class Oodle:
                HZDEditor = bpy.context.scene.HZDEditor
            
                # _local_path = Path(__file__).absolute().parent
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
            with open(HZDEditor.GamePath + "Packed_DX12\\" + self.DesiredArchive, 'rb') as f, open(ExtractedFilePath, 'wb') as w:
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
        for binArchive in ['Initial.bin','Remainder.bin','DLC1.bin']:
            # say("Searching for "+filePath+" in "+binArchive)
            with open(HZDEditor.GamePath + "Packed_DX12\\" + binArchive, 'rb') as f:
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
                    if file.hash == DesiredHash:
                        # file.print()
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

def ClearProperties(self,context):
    HZDEditor = bpy.context.scene.HZDEditor
    HZDEditor.HZDAbsPath = bpy.path.abspath(HZDEditor.HZDPath)
    HZDEditor.GameAbsPath = bpy.path.abspath(HZDEditor.GamePath)
    HZDEditor.WorkAbsPath = bpy.path.abspath(HZDEditor.WorkPath)
    HZDEditor.SkeletonAbsPath = bpy.path.abspath(HZDEditor.SkeletonPath)
    HZDEditor.SkeletonName = "Unknown Skeleton: Import Skeleton to set."
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
    def uuid(f):
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
    def int64(f):
        b = f.read(8)
        i = unpack('<Q',b)[0]
        return i
    @staticmethod
    def string(f,length):
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def path(f, length):
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
class BytePacker:
    @staticmethod
    def int8(v):
        return pack('<b', v)

    @staticmethod
    def uint8(v):
        return pack('<B', v)

    @staticmethod
    def int16(v):
        return pack('<h', v)
    @staticmethod
    def uint16(v):
        return pack('<H', v)

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
    def int64(v):
        return pack('<Q', v)

    @staticmethod
    def float(v):
        return pack('<f', v)

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
    r = ByteReader()
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
def ParseVertex(f,stride,half=False,boneCount=0):
    r = ByteReader()
    startOffset = f.tell()
    endOffset = startOffset+stride

    #Positions
    if half:
        coLength = 8
        x = r.float16(f)
        y = r.float16(f)
        z = r.float16(f)
        f.seek(2,1)
    else:
        coLength = 12
        x = r.float(f)
        y = r.float(f)
        z = r.float(f)
    #Bone Indices
    boneIndices = []
    noWeight = False
    if boneCount >= 256:
        bic = (stride - coLength)/3
        if bic - int(bic) != 0:
            if stride == 32:
                bic = 8
            else:
                bic = (stride - coLength)/2
                noWeight = True
        bint16 = True
    else:
        bic = (stride - coLength)/2 #bone indices count
        bint16 = False
    lastbi =-1
    for i in range(int(bic)):
        if bint16:
            bi = r.uint16(f)
        else:
            bi = r.uint8(f)
        if bi == lastbi:
            pass
        else:
            boneIndices.append(bi)
        lastbi = bi

    #Vertex Weights
    boneWeights = []
    for i,b in enumerate(boneIndices):
        if noWeight:
            boneWeights.append(1.0)
        else:
            bw = r.uint8(f)
            if bw == 0:
                    vw = 1 - sum(boneWeights)
            else:
                vw = bw / 255
                # vw = vw*(-1)+1
            boneWeights.append(vw)
    if not noWeight:
        f.seek(endOffset)
    return (x,y,z),boneIndices,boneWeights
def ParseNormals(f):
    r = ByteReader()
    # Normal Vector
    nx = r.float(f)
    ny = r.float(f)
    nz = r.float(f)
    # Tangent Vector
    tx = r.float(f)
    ty = r.float(f)
    tz = r.float(f)
    # Flip
    flip = r.float(f)
    return (nx,ny,nz), [tx,ty,tz], flip
def ParseColor(f):
    r = ByteReader()
    rc = r.uint8(f)/255
    gc = r.uint8(f)/255
    bc = r.uint8(f)/255
    ac = r.uint8(f)/255
    return (rc,gc,bc,ac)
def ParseUV(f):
    r = ByteReader()
    u = r.float16(f)
    v = r.float16(f)
    # u = r.int16(f)
    # v = r.int16(f)
    # u2 = r.float16(f)
    # v2 = r.float16(f)
    # u3 = r.float16(f)
    # v3 = r.float16(f)
    # u = u/65536
    # v = v/65536
    return (u,v)
def ParseFaces(f):
    r = ByteReader()
    v1 = r.uint16(f)
    v2 = r.uint16(f)
    v3 = r.uint16(f)
    face = (v1,v2,v3)
    return face

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
    SkeletonName: bpy.props.StringProperty(name="Skeleton Name")

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


def ImportMesh(isGroup,Index,LODIndex,BlockIndex):
    r = ByteReader()
    # print(isGroup,Index,LODIndex,BlockIndex)
    if isGroup:
        md = asset.LODGroups[Index].LODList[LODIndex].meshBlockList[BlockIndex]
        meshName = asset.LODGroups[Index].LODList[LODIndex].meshNameBlock.name
    else:
        md = asset.LODObjects[Index].LODList[LODIndex].meshBlockList[BlockIndex]
        meshName = asset.LODObjects[Index].LODList[LODIndex].meshNameBlock.name
    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath
    stream = core+".stream"
    coresize = os.path.getsize(core)

    say("\nImporting : "+str(BlockIndex)+"_"+meshName)

    # CREATE COLLECTION TREE #####################
    if bpy.context.scene.collection.children.find(meshName[:-1]) >= 0:
        assetCollection = bpy.context.scene.collection.children[meshName[:-1]]
    else:
        assetCollection = bpy.context.blend_data.collections.new(name=meshName[:-1])
        bpy.context.scene.collection.children.link(assetCollection)

    if assetCollection.children.find("LOD " + meshName[-1:].capitalize()) >= 0:
        lodCollection = assetCollection.children["LOD " + meshName[-1:].capitalize()]
    else:
        lodCollection = bpy.context.blend_data.collections.new(name="LOD " + meshName[-1:].capitalize())
        assetCollection.children.link(lodCollection)

    # Attach to Armature #####################
    armature = None
    for o in assetCollection.objects:
        if type(o.data) == bpy.types.Armature:
            if o.data.name[0:20] == str(ArchiveManager.get_file_hash(asset.LODGroups[0].LODList[0].meshNameBlock.skeletonPath)):
                armature = o
    if armature is None:
        armature = CreateSkeleton()


    boneCount = len(armature.data.bones)

    with open(stream,'rb') as f:
        #VERTICES
        vList = [] #Vertices
        biList = [] #Bone Indices
        bwList = [] #Bone Weights
        vb = md.vertexBlock


        f.seek(vb.vertexStream.dataOffset)
        for n in range(vb.vertexCount):
            vertex,vBoneIndices,vBoneWeights = ParseVertex(f,vb.vertexStream.stride,vb.coHalf,boneCount)
            vList.append(vertex)
            biList.append(vBoneIndices)
            bwList.append(vBoneWeights)
        vertCount = len(vList)

        #TRIANGLES
        fList = []
        fb = md.faceBlock
        f.seek(fb.faceDataOffset)
        for n in range(int(fb.indexCount/3)):
            face = ParseFaces(f)
            if face[0] < vertCount and face[1] < vertCount and face[2] < vertCount:
                fList.append(face)
        # print(len(fList))
        #NORMALS
        if vb.normalsStream:
            nList = []
            tList = []
            flipList = []
            if vb.realOffsets:
                f.seek(vb.normalsStream.dataOffset)
            else:
                f.seek(vb.normalsStream.dataOffset+vb.vertexStream.dataSize)
            for n in range(vb.vertexCount):
                normal,tangent,flip = ParseNormals(f)
                nList.append(normal)
                tList.append(tangent)
                flipList.append(flip)

        #UV AND COLOR
        uList = []
        uList2 = []
        cList = []

        if vb.realOffsets:
            f.seek(vb.uvStream.dataOffset)
        else:
            f.seek(vb.uvStream.dataOffset+vb.vertexStream.dataSize)
            if vb.normalsStream:
                f.seek(vb.normalsStream.dataSize,1)
        for i in range(vb.vertexCount):
            if vb.hasVertexColor:
                vc = ParseColor(f)
                cList.append(vc)
            uv = ParseUV(f)
            uList.append(uv)
            if vb.hasTwoUV:
                uv2 = ParseUV(f)
                uList2.append(uv2)
    # BUILD MESH ////////////////////////////////////////////////////
    mesh = bpy.data.meshes.new(meshName+"_MESH")
    obj = bpy.data.objects.new(str(BlockIndex)+"_"+meshName,mesh)
    lodCollection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # VERTICES ####################################
    for v in vList:
        vert = bm.verts.new()
        vert.co = v
    bm.verts.ensure_lookup_table()
    # TRIANGLES ####################################
    for f in fList:
        if f[0] != f[1] != f[2] != f[0]:
            fvs = []
            for i in f:
                fv = bm.verts[i]
                fvs.append(fv)
            bface = bm.faces.new(fvs)
            bface.smooth = True

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()  # prevents -1 indices, ensure_lookup_table didn't seem to work here
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # UV AND VERTEX COLOR ####################################
    color_layer = bm.loops.layers.color.new("Color")
    uv_layer = bm.loops.layers.uv.new("UV")
    if vb.hasTwoUV:
        uv_layer2 = bm.loops.layers.uv.new("UV2")
    for findex,face in enumerate(bm.faces):
        for lindex, loop in enumerate(face.loops):
            loop[uv_layer].uv = uList[loop.vert.index]
            if vb.hasVertexColor:
                loop[color_layer] = cList[loop.vert.index]
            if vb.hasTwoUV:
                loop[uv_layer2].uv = uList2[loop.vert.index]

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # NORMALS #######################################

    if vb.normalsStream:
        mesh.use_auto_smooth = True
        mesh.normals_split_custom_set_from_vertices(nList)


    # VERTEX GROUPS ####################################
    SkeletonPath = HZDEditor.SkeletonAbsPath
    CoreBones = []

    with open(SkeletonPath, 'rb') as f: #Get every bone name because blender's bone indices don't match HZD
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
            CoreBones.append(boneName)

    for bone in CoreBones: #Create vertex Groups
        obj.vertex_groups.new(name=bone)
    # deform_layer = bm.verts.layers.deform.new()



    for v in mesh.vertices:
        vindex = v.index

        for index, boneindex in enumerate(biList[vindex]):

            index = index - 1
            if index == -1:
                index = len(biList[vindex]) - 1
            if len(CoreBones)>boneindex:
                coreBone = CoreBones[boneindex] #Using the order from the skeleton file, boneindex gives the correct bone name.
                weight = bwList[vindex][index] #BoneWeight for the current vertex(vindex), index gives the
                obj.vertex_groups[coreBone].add([vindex], weight, "ADD")
            else:
                raise Exception("Vertex wasn't parsed correctly.{v} is not a bone".format(v=boneindex))


    # Attach to Armature ##################
    obj.modifiers.new(name='Skeleton', type='ARMATURE')
    obj.modifiers['Skeleton'].object = armature
    obj.parent = armature
    # Check if armature is in the asset collection
    if assetCollection.objects.find(armature.name) == -1:
        armature.users_collection[0].objects.unlink(armature)
        assetCollection.objects.link(armature)

    if HZDEditor.ExtractTextures:
        matblock = asset.LODGroups[Index].LODList[LODIndex].materialBlockList[BlockIndex]
        CreateMaterial(obj,matblock,meshName)

def ExtractAsset(assetPath):

    AM = ArchiveManager()
    #Extract Asset

    assetFile = AM.FindFile(assetPath)
    filePath = AM.ExtractFile(assetFile,assetPath,False)

    assetStreamFile = AM.FindFile(assetPath+".stream")
    fileStreamPath = AM.ExtractFile(assetStreamFile, assetPath+".stream", False)

    HZDEditor = bpy.context.scene.HZDEditor
    HZDEditor.HZDPath = filePath

    ReadCoreFile()
    skeletonFile = asset.LODGroups[0].LODList[0].meshNameBlock.skeletonPath
    say(skeletonFile)

    assetSkeletonFile = AM.FindFile(skeletonFile + ".core")
    fileSkeletonPath = AM.ExtractFile(assetSkeletonFile, skeletonFile + ".core", False)
    HZDEditor.SkeletonPath = fileSkeletonPath
    return

def CreateNormalConverterGroup():
    # Based on mithkr's node group
    if bpy.data.node_groups.find("HZD Normal Map Converter") != -1:
        return
    group = bpy.data.node_groups.new("HZD Normal Map Converter","ShaderNodeTree")
    group.inputs.new("NodeSocketFloat","Strength")
    group.inputs["Strength"].default_value = 1.0
    group.inputs.new("NodeSocketColor","Image")
    group.inputs["Image"].default_value = (0.5,0.5,1.0,1.0)
    group.outputs.new("NodeSocketVector","Normal")

    groupIn = group.nodes.new("NodeGroupInput")
    groupIn.location = -900,20
    groupOut = group.nodes.new("NodeGroupOutput")
    groupOut.location = 1285,100

    #NormalMap
    normalMap = group.nodes.new("ShaderNodeNormalMap")
    normalMap.location = 1079,88
    group.links.new(normalMap.inputs[0],groupIn.outputs[0])
    group.links.new(groupOut.inputs[0],normalMap.outputs[0])

    #Combine RGB
    combineRGB = group.nodes.new('ShaderNodeCombineRGB')
    combineRGB.location = 905,-40
    group.links.new(normalMap.inputs[1],combineRGB.outputs[0])

    #Separate RGB
    separateRGB = group.nodes.new('ShaderNodeSeparateRGB')
    separateRGB.location = -720, -60
    group.links.new(combineRGB.inputs[0],separateRGB.outputs[0])
    group.links.new(separateRGB.inputs[0],groupIn.outputs[1])

    #InvertGreen
    invertGreen = group.nodes.new('ShaderNodeInvert')
    invertGreen.location = -475, -110
    group.links.new(combineRGB.inputs[1],invertGreen.outputs[0])
    group.links.new(invertGreen.inputs[1],separateRGB.outputs[1])

    #SquareRoot
    squareRoot = group.nodes.new('ShaderNodeMath')
    squareRoot.operation = "SQRT"
    squareRoot.location = 725, -140
    group.links.new(combineRGB.inputs[2],squareRoot.outputs[0])

    #Clamp
    clamp = group.nodes.new('ShaderNodeClamp')
    clamp.location = 545, -220
    group.links.new(squareRoot.inputs[0],clamp.outputs[0])

    #Invert
    invert = group.nodes.new('ShaderNodeInvert')
    invert.location = 385, -220
    group.links.new(clamp.inputs[0], invert.outputs[0])

    #Dot Product
    dot = group.nodes.new('ShaderNodeVectorMath')
    dot.operation = "DOT_PRODUCT"
    dot.location = 225,-220
    group.links.new(invert.inputs[1],dot.outputs[1])

    #Combine XYZ
    combineXYZ = group.nodes.new('ShaderNodeCombineXYZ')
    combineXYZ.location = 60, -220
    group.links.new(dot.inputs[0],combineXYZ.outputs[0])
    group.links.new(dot.inputs[1],combineXYZ.outputs[0])

    #Subtract X
    subtractX = group.nodes.new('ShaderNodeMath')
    subtractX.operation = 'SUBTRACT'
    subtractX.inputs[1].default_value = 1.0
    subtractX.location = -120, -180
    group.links.new(combineXYZ.inputs[0],subtractX.outputs[0])

    # Subtract Y
    subtractY = group.nodes.new('ShaderNodeMath')
    subtractY.operation = 'SUBTRACT'
    subtractY.inputs[1].default_value = 1.0
    subtractY.location = -120, -340
    group.links.new(combineXYZ.inputs[1], subtractY.outputs[0])

    # Multiply X
    multiplyX = group.nodes.new('ShaderNodeMath')
    multiplyX.operation = 'MULTIPLY'
    multiplyX.inputs[1].default_value = 2.0
    multiplyX.location = -280, -180
    group.links.new(subtractX.inputs[0], multiplyX.outputs[0])
    group.links.new(invertGreen.outputs[0],multiplyX.inputs[0])

    # Multiply Y
    multiplyY = group.nodes.new('ShaderNodeMath')
    multiplyY.operation = 'MULTIPLY'
    multiplyY.inputs[1].default_value = 2.0
    multiplyY.location = -280, -340
    group.links.new(subtractY.inputs[0], multiplyY.outputs[0])
    group.links.new(multiplyY.inputs[0],separateRGB.outputs[1])
def ExtractTexture(outWorkspace,texPath):
    texAs = None
    def BuildDDSHeader(tex:Texture) -> bytes:
        r = BytePacker
        data = bytes()
        flags = b'\x07\x10\x00\x00'
        data += flags

        data += r.uint32(tex.height)
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
            outImage = outPath.with_name(t.name + ".dds")
            if os.path.exists(outImage):
                textureFiles.append(outImage)
            else:
                streamData = bytes()
                if t.streamSize32 > 0:
                    # Check if .stream was already there
                    if os.path.exists(outWorkspace + texPath+".core.stream"):
                        streamFilePath = outWorkspace + texPath+".core.stream"
                    else:
                        streamFileEntry = AM.FindFile(texPath+".core.stream")
                        streamFilePath = AM.ExtractFile(streamFileEntry,texPath,True)
                    with open(streamFilePath,'rb') as s:
                        s.seek(t.streamOffset)
                        streamData = s.read(t.streamSize64)

                if os.path.exists(outImage):
                    textureFiles.append(outImage)
                else:
                    with open(outImage,'wb') as w:
                        w.write(BuildDDSHeader(t))
                        w.write(streamData)
                        w.write(t.thumbnail)
                        textureFiles.append(outImage)

    textureFiles = []
    AM = ArchiveManager()

    if os.path.exists(outWorkspace+texPath+".core"):
        ParseTexture(outWorkspace+texPath+".core")
    else:
        #Extract Core
        texFileEntry = AM.FindFile(texPath+".core")
        filePath = AM.ExtractFile(texFileEntry,texPath,False)

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
                 "Mask":"Float",  # 8
                 "Mask_Alpha":"Float",  # 9
                 "Incandescence":"Float",  # 10
                 "Translucency_Diffusion":"Float",  # 11
                 "Translucency_Amount":"Float",  # 12
                 "Misc_01":"Float",  # 13
                 "Count":"Float"}  # 14


    if bpy.data.materials.find(str(matblock.shaderName+"___"+meshName)) == -1:
        mat = bpy.data.materials.new(name=str(matblock.shaderName+"___"+meshName))
        obj.data.materials.append(mat)
        mat.use_nodes = True
        for i,t in enumerate(matblock.uniqueTextures):
            images,texAsset = ExtractTexture(HZDEditor.WorkAbsPath, t)

            if texAsset.texSet is not None:

                # Create Node Group
                imageName = t.split('/')
                imageName= imageName[len(imageName) - 1]
                texSetGroup = bpy.data.node_groups.new(imageName,"ShaderNodeTree")
                texSetGroup_output = texSetGroup.nodes.new("NodeGroupOutput")

                for ii, setT in enumerate(texAsset.texSet.textures):
                    # Create Image Node
                    texNode = texSetGroup.nodes.new('ShaderNodeTexImage')
                    texNode.name = t
                    texNode.label = imageName
                    texNode.location = -400, -250 * ii + len(images) * 125
                    bpy.data.images.load(str(images[ii]))
                    texNode.image = bpy.data.images[images[ii].name]
                    texNode.image.colorspace_settings.name = "Non-Color"
                    print(imageName)


                    # RGB CHANNEL OUTPUT
                    if all(cha.usageType == setT.channelTypes[0].usageType for cha in setT.channelTypes[0:3]):
                        # no need to break the color
                        if setT.channelTypes[0].usageType == "Color":
                            texNode.image.colorspace_settings.name = "sRGB"
                        outputType = UsageType_ValueMap[setT.channelTypes[0].usageType]
                        texSetGroup.outputs.new("NodeSocket"+outputType, setT.channelTypes[0].usageType)
                        texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs)-2],texNode.outputs[0])

                    elif all(cha.usageType == "Normal" for cha in setT.channelTypes[0:2]):
                        # Normal Map
                        CreateNormalConverterGroup()
                        normalConverter = texSetGroup.nodes.new("ShaderNodeGroup")
                        normalConverter.node_tree = bpy.data.node_groups["HZD Normal Map Converter"]
                        texNode.location = texNode.location[0] - 400, texNode.location[1]  # move to the left
                        normalConverter.location = texNode.location[0] + 400, texNode.location[1]
                        texSetGroup.outputs.new('NodeSocketVector',"Normal")
                        texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs)-2],normalConverter.outputs[0])
                        texSetGroup.links.new(normalConverter.inputs[1],texNode.outputs[0])
                        #Blue channel
                        sepRGBNode = texSetGroup.nodes.new("ShaderNodeSeparateRGB")
                        sepRGBNode.location = texNode.location[0] + 400, texNode.location[1]-140
                        texSetGroup.outputs.new('NodeSocketFloat', setT.channelTypes[2].usageType)

                        texSetGroup.links.new(sepRGBNode.inputs[0],texNode.outputs[0])
                        if setT.channelTypes[2].usageType == "Roughness":
                            invert = texSetGroup.nodes.new("ShaderNodeInvert")
                            invert.location = texNode.location[0] + 400, texNode.location[1]-280
                            texSetGroup.links.new(invert.inputs[1],sepRGBNode.outputs[2])
                            texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs) - 2],invert.outputs[0])
                        else:
                            texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs) - 2],sepRGBNode.outputs[2])

                    else:
                        texNode.location = texNode.location[0] - 400, texNode.location[1]  # move to the left
                        sepRGBNode = texSetGroup.nodes.new("ShaderNodeSeparateRGB")
                        sepRGBNode.location = texNode.location[0] + 400, texNode.location[1]
                        for ic,cha in enumerate(setT.channelTypes[0:3]):
                            if cha.usageType == "Invalid":
                                pass
                            elif cha.usageType == "Roughness":
                                invert = texSetGroup.nodes.new("ShaderNodeInvert")
                                invert.location = texNode.location[0] + 400, texNode.location[1] - 280
                                texSetGroup.links.new(invert.inputs[1], sepRGBNode.outputs[ic])
                                texSetGroup.outputs.new("NodeSocketFloat", cha.usageType)
                                texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs) - 2],sepRGBNode.outputs[ic])
                            else:
                                # we gotta separate RGB
                                texSetGroup.outputs.new("NodeSocketFloat", cha.usageType)

                                texSetGroup.links.new(sepRGBNode.inputs[0], texNode.outputs[0])
                                texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs)-2],sepRGBNode.outputs[ic])



                    # ALPHA CHANNEL OUTPUT
                    if setT.channelTypes[3].usageType in ("Invalid","Normal"):
                        pass
                    else:
                        texSetGroup.outputs.new("NodeSocketFloat", setT.channelTypes[3].usageType)
                        texSetGroup.links.new(texSetGroup_output.inputs[len(texSetGroup_output.inputs)-2],texNode.outputs[1])

                shaderGroup = mat.node_tree.nodes.new("ShaderNodeGroup")
                shaderGroup.node_tree = bpy.data.node_groups[texSetGroup.name]
                shaderGroup.location = -400,-300*i+600



            else:
                for ii, image in enumerate(images):
                    texNode = mat.node_tree.nodes.new('ShaderNodeTexImage')
                    texNode.name = t
                    imageName = t.split('/')
                    texNode.label = imageName[len(imageName)-1]
                    texNode.location = -400*ii-400,-300*i+600
                    bpy.data.images.load(str(image))
                    texNode.image = bpy.data.images[image.name]

    else:
        say("Material already exists")
        mat = bpy.data.materials[matblock.shaderName+"___"+meshName]
        obj.data.materials.append(mat)

def CreateSkeleton():
    r = ByteReader()
    HZDEditor = bpy.context.scene.HZDEditor
    SkeletonPath = HZDEditor.SkeletonAbsPath
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

    armatureName = str(ArchiveManager.get_file_hash(asset.LODGroups[0].LODList[0].meshNameBlock.skeletonPath))
    armature = bpy.data.armatures.new(armatureName)
    obj = bpy.data.objects.new(sktName, armature)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    HZDEditor.SkeletonName = obj.name

    bpy.ops.object.mode_set(mode="EDIT")

    for i,b in enumerate(Bones):
        bone = armature.edit_bones.new(b)
        bone.parent = armature.edit_bones[ParentIndices[i]]
        # print(bone.parent)
        bone.tail = mathutils.Vector([0,0,1])

    for b in BoneMatrices:
        bone = armature.edit_bones[b]
        bone.tail = mathutils.Vector([0,0,1])
        bone.transform(BoneMatrices[b])

    bpy.ops.object.mode_set(mode='OBJECT')

    return obj

def PackVertex(f,vertex,stride,half=False,boneCount=0):
    p = BytePacker()
    bVertex = b''
    x = vertex.co[0]
    y = vertex.co[1]
    z = vertex.co[2]
    if half:
        coLength = 8
        bVertex += p.float16(x)
        bVertex += p.float16(y)
        bVertex += p.float16(z)
        bVertex += b'\x00\x3C'
    else:
        coLength = 12
        bVertex += p.float(x)
        bVertex += p.float(y)
        bVertex += p.float(z)
    # Bone Indices
    if boneCount >= 256:
        bic = (stride - coLength) / 3
        if bic - int(bic) != 0:
            if stride == 32:
                bic = 8
            else:
                bic = (stride - coLength) / 2
        bint16 = True
        bic = int(bic)
    else:
        bic = int((stride - coLength) / 2)  # bone indices count
        bint16 = False
    # Gather bone groups
    groupsweights = {}
    for vg in vertex.groups:
        if vg.weight > 0.0:
            groupsweights[vg.group] = vg.weight
    # print(groupsweights)
    #Normalize
    totalweight = 0.0
    for gw in groupsweights:
        totalweight += groupsweights[gw]
    if totalweight == 0:
        raise Exception("Vertex {v} has no weight".format(v=vertex.index))
    normalizer = 1/totalweight
    for gw in groupsweights:
        groupsweights[gw] *= normalizer
    #Sort Weights
    sortedweights = sorted(groupsweights.items(),key=operator.itemgetter(1),reverse=True)
    #Truncate Weights
    truncweights = sortedweights[0:bic]
    boneRepeat = bic - len(truncweights)
    # print(len(truncweights))
    for i, bw in enumerate(truncweights):
        if bint16:
            bVertex += p.uint16(bw[0])
        else:
            bVertex += p.uint8(bw[0])
    # print(len(bVertex))
    for i  in range(boneRepeat):
        if bint16:
            bVertex += p.uint16(truncweights[len(truncweights)-1][0])
        else:
            bVertex += p.uint8(truncweights[len(truncweights) - 1][0])
    # print(len(bVertex))
    for i,bw in enumerate(truncweights):
        if i == 0:
            pass #the biggest weight goes as the remainder
        else:
            # print(bw[1],bw[1]*255,int(round(bw[1]*255)))
            bVertex += p.uint8(int(round(bw[1]*255)))
    # print(len(bVertex))
    #Fill remainder
    if bint16:
        vbLength = coLength+(bic*2)+bic
    else:
        vbLength = coLength + bic + bic
    # print(vbLength,len(bVertex))
    for b in range(len(bVertex),stride):
        bVertex += b'\x00'
    # print(bic, boneRepeat,len(bVertex))
    if len(bVertex) == stride:
        f.write(bVertex)
    else:
        raise Exception("Vertex bytes not expected length:{v} instead of {e}".format(v=len(bVertex),e=stride))
def PackNormal(f,ntb,stride):
    p = BytePacker()
    bNormals = b''
    nx = ntb[0][0]
    ny = ntb[0][1]
    nz = ntb[0][2]
    bNormals += p.float(nx)
    bNormals += p.float(ny)
    bNormals += p.float(nz)
    if stride == 28:
        tx = ntb[1][0]
        ty = ntb[1][1]
        tz = ntb[1][2]
        bs = ntb[2]
        bNormals += p.float(tx)
        bNormals += p.float(ty)
        bNormals += p.float(tz)
        bNormals += p.float(bs)

    f.write(bNormals)
def PackUVs(f,uv):
    p = BytePacker()
    bUV = b''
    if uv[2] is not None: #VERTEX COLOR
        # print("Has Vertex Color",uv)
        bUV += p.uint8(int(uv[2][0]*255))
        bUV += p.uint8(int(uv[2][1]*255))
        bUV += p.uint8(int(uv[2][2]*255))
        bUV += p.uint8(int(uv[2][3]*255))
    bUV += p.float16(uv[0][0]) #UV
    bUV += p.float16(uv[0][1])
    if uv[1] is not None: #UV2
        # print("Has 2 UVs",uv)
        bUV += p.float16(uv[1][0])
        bUV += p.float16(uv[1][1])
    f.write(bUV)
def PackFace(f,face):
    p = BytePacker()
    bFace = b''
    v1 = face.vertices[0]
    v2 = face.vertices[1]
    v3 = face.vertices[2]
    bFace += p.uint16(v1)
    bFace += p.uint16(v2)
    bFace += p.uint16(v3)
    f.write(bFace)

def ExportMesh(isGroup,Index,LODIndex,BlockIndex):
    r = ByteReader()
    p = BytePacker()
    print(isGroup, Index, LODIndex, BlockIndex)
    if isGroup:
        lod = asset.LODGroups[Index].LODList[LODIndex]
        md = lod.meshBlockList[BlockIndex]
        meshName = lod.meshNameBlock.name
    else:
        lod = asset.LODObjects[Index].LODList[LODIndex]
        md = lod.meshBlockList[BlockIndex]
        meshName = lod.meshNameBlock.name
    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath
    stream = core + ".stream"
    coresize = os.path.getsize(core)
    vb = md.vertexBlock
    fb = md.faceBlock

    objectName = str(BlockIndex)+"_"+meshName
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
        CopyFile(f, w, 0, md.vertexBlock.vertexStream.dataOffset) #Copy the source file up to the vertex position

        #Vertices

        newCoVOffset = w.tell() #New offset of our vertex stream
        for v in editedMesh.vertices:
            PackVertex(w,v,vb.vertexStream.stride,vb.coHalf,boneCount)
        FillChunk(w)
        newCoVSize = w.tell()-newCoVOffset

        #Normals
        if vb.normalsStream:
            newCoNOffset = w.tell()
            editedMesh.calc_tangents()
            NTB = [((1.0,0.0,0.0),(0.0,1.0,0.0),0.0)] * len(editedMesh.vertices) #Normal Tangent Bi-tangent
            #Get Normals
            for l in editedMesh.loops:
                if l.bitangent_sign == -1:
                    flip = 1.0
                else:
                    flip = 0.0
                NTB[l.vertex_index] = (l.normal, l.tangent, flip)
            #Write Normals
            for n in NTB:
                PackNormal(w,n,vb.normalsStream.stride)
            FillChunk(w)
            newCoNSize = w.tell()-newCoNOffset

        #UVs
        newCoUOffset = w.tell()
        UVs = [((0.0,0.0),(0.0,0.0),(0,0,0,0))] * len(editedMesh.vertices)
        bm = bmesh.new()
        bm.from_mesh(editedMesh)
        bm.faces.ensure_lookup_table()
        #Get UVs and Color
        for bface in bm.faces:
            for loop in bface.loops:
                vertUV = [None,None,None] #UV1,UV2,Color

                uv1 = loop[bm.loops.layers.uv[0]].uv
                vertUV[0] = uv1
                if vb.hasTwoUV:
                    if len(editedMesh.uv_layers) == 2:
                        uv2 = loop[bm.loops.layers.uv[1]].uv
                        vertUV[1] = uv2
                    else:
                        raise Exception("Mesh Block is expecting 2 UV Layers")
                if vb.hasVertexColor:
                    if len(editedMesh.vertex_colors) == 1:
                        vcolor = loop[bm.loops.layers.color[0]]
                        vertUV[2] = vcolor
                    else:
                        raise Exception("Mesh Block is expecting 1 Vertex Color layer")
                UVs[loop.vert.index] = vertUV
        #Write UVs
        for uvindex,uv in enumerate(UVs):
            PackUVs(w,uv)
        FillChunk(w)
        newCoUSize = w.tell()-newCoUOffset

        #Faces
        newCoFOffset = w.tell()
        for poly in editedMesh.polygons:
            PackFace(w,poly)
        FillChunk(w)
        newCoFSize = w.tell()-newCoFOffset

        endOffset = md.faceBlock.faceDataOffset + md.faceBlock.faceDataSize
        CopyFile(f,w,endOffset,streamsize-endOffset) #Copy the source file to the end.

    # Write Core
    with open(sourcecore, 'rb') as f, open(core+"TMP",'wb+') as w:
        CopyFile(f,w,0,coresize) #full copy of source core
        # WRITE NEW VALUES FOR THE CURRENT MESH BLOCK
        #Vertex Counts
        w.seek(vb.posVCount)
        w.write(p.int32(len(editedMesh.vertices)))
        w.seek(lod.meshBlockInfo.meshInfos[BlockIndex].posVCount)
        w.write(p.int32(len(editedMesh.vertices)))
        #Vertex
        w.seek(vb.vertexStream.posOffset)
        w.write(p.int64(newCoVOffset))
        w.seek(vb.vertexStream.posSize)
        w.write(p.int64(newCoVSize))
        #Edges

        #Normals
        if vb.normalsStream:
            w.seek(vb.normalsStream.posOffset)
            if vb.realOffsets:
                w.write(p.int64(newCoNOffset))
            else:
                w.write(p.int64(newCoVOffset))
            w.seek(vb.normalsStream.posSize)
            w.write(p.int64(newCoNSize))
        #UVs
        w.seek(vb.uvStream.posOffset)
        if vb.realOffsets:
            w.write(p.int64(newCoUOffset))
        else:
            w.write(p.int64(newCoVOffset))
        w.seek(vb.uvStream.posSize)
        w.write(p.int64(newCoUSize))
        #Faces
        w.seek(fb.posIndexCount)
        w.write(p.int32(len(editedMesh.polygons)*3))
        w.seek(md.stuffBlock.posOffset)
        w.write(p.int32(len(editedMesh.polygons)*3))
        w.seek(fb.posOffset)
        w.write(p.int64(newCoFOffset))
        w.seek(fb.posSize)
        w.write(p.int64(newCoFSize))

        # the place where new mesh block ends minus where it ended before
        DiffOff = (newCoFOffset + newCoFSize) - (fb.faceDataOffset + fb.faceDataSize)
        # print(DiffOff)
        def AddDiff(pos,diff=DiffOff):
            if pos != 0:
                w.seek(pos)
                oldOffset = r.int64(w)
                w.seek(pos)
                w.write(p.int64(oldOffset+diff))
        def mdDiff(xmd):
            # Vertex
            if xmd.vertexBlock.vertexStream:
                AddDiff(xmd.vertexBlock.vertexStream.posOffset)
                # print(objectName,"  Vertex  ",xmd.vertexBlock.vertexStream.posOffset)
            # Edge
            if xmd.edgeBlock:
                AddDiff(xmd.edgeBlock.posOffset)
            # Normals
            if xmd.vertexBlock.normalsStream:
                AddDiff(xmd.vertexBlock.normalsStream.posOffset)
            # UV
            if xmd.vertexBlock.uvStream:
                AddDiff(xmd.vertexBlock.uvStream.posOffset)
            # Faces
            if xmd.faceBlock:
                AddDiff(xmd.faceBlock.posOffset)

        #Group are after Objects, so no need to add to objects.
        if isGroup:
            # Remaining mesh blocks of current Lod
            for md in asset.LODGroups[Index].LODList[LODIndex].meshBlockList[BlockIndex + 1:]:
                mdDiff(md)
            # The following LODs
            for l in asset.LODGroups[Index].LODList[LODIndex+1:]:
                for md in l.meshBlockList:
                    mdDiff(md)
            #just in case there are other groups
            for g in asset.LODGroups[Index+1:]:
                for l in g.LODList:
                    for md in l.meshBlockList:
                        mdDiff(md)

        else:
            # Remaining mesh blocks of current Lod
            for md in asset.LODObjects[Index].LODList[LODIndex].meshBlockList[BlockIndex + 1:]:
                mdDiff(md)
            # The following LODs
            for l in asset.LODObjects[Index].LODList[LODIndex + 1:]:
                for md in l.meshBlockList:
                    mdDiff(md)
            # the other objects
            for o in asset.LODObjects[Index + 1:]:
                for l in o.LODList:
                    for md in l.meshBlockList:
                        mdDiff(md)
            # do every md in every lod of every LODGroup
            for g in asset.LODGroups:
                for l in g.LODList:
                    for md in l.meshBlockList:
                        mdDiff(md)
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
    p = BytePacker()
    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath

    with open(core,'rb+') as w:
        w.seek(asset.LODGroups[Index].blockStartOffset)
        # print(w.tell())
        r = ByteReader()
        w.seek(16, 1)
        r.hashtext(w)
        w.seek(24, 1)
        w.seek(8, 1)
        w.seek(4,1)
        w.seek(16, 1)
        LODCount = r.int32(w)
        # print(w.tell())
        for i in range(LODCount):
            w.seek(17, 1)
            w.write(p.float(HZDEditor["LodDistance" + str(i)]))
            # print(w.tell())



class Asset:
    def __init__(self):
        self.LODGroups = []
        self.LODObjects = []
        self.meshBlocks = []

    # def FindMeshes(self):
    #     for lg in self.LODGroups:
    #         for l in lg.LODList:
    #             for m in l.meshBlockList:
    #                 self.meshBlocks.append(m)
    #     for lo in self.LODObjects:
    #         for l in lo.LODList:
    #             for m in l.meshBlockList:
    #                 self.meshBlocks.append(m)

BlockIDs = {"MeshBlockInfo":4980347625154103665,
            "MeshNameBlock":10982056603708398958,
            "SkeletonBlock":232082505300933932,
            "VertexBlock":13522917709279820436,
            "FaceBlock":12198706699739407665,
            "ShaderBlock":12029122079492233037,
            }
asset = Asset()

class DataBlock:
    def __init__(self,f,expectedID=0):
        r = ByteReader()

        self.expectedID = expectedID
        self.ID = r.int64(f)
        if expectedID != 0:
            if self.ID != self.expectedID:
                raise Exception("%s  --  Invalid Block ID: got %d expected %d"%(self.__class__.__name__,self.ID ,self.expectedID))
        self.size = r.int32(f)
        self.blockStartOffset = f.tell()
        # print(self.__class__.__name__)
        # print("ID = ",self.ID,"\n","Size = ",self.size,"\nStart = ",self.blockStartOffset)
    def EndBlock(self,f):
        f.seek(self.blockStartOffset + self.size)

class TextureAsset:
    def __init__(self,f):
        self.textures = []
        self.texSet = None
        r = ByteReader
        ID = r.int64(f)
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

        f.seek(16, 1)
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
    def __init__(self,f):
        super().__init__(f)
        r = ByteReader
        f.seek(16,1)
        self.name = r.hashtext(f)
        f.seek(2,1)
        width = r.uint16(f)
        self.width = width & 0x3FFF
        height = r.uint16(f)
        self.height = height & 0x3FFF
        f.seek(2,1)
        f.seek(1,1)
        self.format : Texture.PixelFormat = self.PixelFormat(r.uint8(f))
        f.seek(2,1)
        f.seek(20,1)
        self.imageChunkSize= r.int32(f)
        self.thumbnailLength = r.int32(f)
        self.streamSize32 = r.int32(f)
        if self.streamSize32 > 0:
            self.mipCount = r.int32(f)
            pathLength = r.int32(f)
            self.streamPath = r.path(f,pathLength) [6:] #remove "cache:"
            self.streamOffset = r.int64(f)
            self.streamSize64 = r.int64(f)
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
        self.indexOffset = 0
        self.matrixOffset = 0
        self.matrixCount = matrixCount
        self.boneList = []

        self.ParseBoneData(f)

    def ParseBoneData(self,f):
        r = ByteReader()
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
class SkeletonBlock(DataBlock):
    def __init__(self, f):
        super().__init__(f,232082505300933932)
        self.matrixCount = 0
        self.boneData = None

        self.ParseSkeleton(f)

    def ParseSkeleton(self,f):
        r = ByteReader()

        f.seek(20,1)
        self.matrixCount = r.int32(f)
        self.boneData = BoneData(f,self.matrixCount)

        self.EndBlock(f)

class MeshNameBlock(DataBlock):
    def __init__(self, f):
        super().__init__(f,10982056603708398958)
        self.name = ""
        self.skeletonPath = ""

        self.ParseMeshName(f)

    def ParseMeshName(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.name = r.hashtext(f)
        f.seek(65,1)
        self.skeletonPath = r.hashtext(f)
        self.EndBlock(f)
class MeshInfo:
    def __init__(self,f):
        r = ByteReader()
        f.seek(8,1)
        f.seek(16,1)
        self.posVCount = f.tell()
        self.vertexCount = r.int32(f)
        f.seek(13,1)
class MeshBlockInfo(DataBlock):
    def __init__(self,f):
        super().__init__(f,4980347625154103665)
        self.meshBlockCount = 0
        self.meshInfos = []

        self.ParseMeshBlockInfo(f)

    def ParseMeshBlockInfo(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.meshBlockCount = r.int32(f)
        for i in range(self.meshBlockCount):
            self.meshInfos.append(MeshInfo(f))
        self.EndBlock(f)

class StreamRef:
    def __init__(self,f,blockOffset,blockSize):
        self.blockOffset = blockOffset
        self.blockSize = blockSize
        self.stride = 0
        self.pathLength = 0
        self.streamPath = ""
        self.dataOffset = 0
        self.dataSize = 0
        self.posOffset = 0
        self.posSize = 0

        self.ParseStreamRef(f)

    def ParseStreamRef(self,f):
        r = ByteReader()
        f.seek(4,1)
        self.stride = r.int32(f)
        unknownCount = r.int32(f)
        f.seek(16,1)
        # print(unknownCount)
        for i in range(unknownCount):
            f.seek(4,1)
        self.pathLength = r.int32(f)
        if 12 >= self.pathLength > 2000: # kinda arbitrary numbers, was easier to put 2000 than getting file size.
            f.seek(self.blockOffset+self.blockSize)
        else:
            self.streamPath = r.path(f,self.pathLength)
            self.posOffset = f.tell()
            self.dataOffset = r.int64(f)
            self.posSize = f.tell()
            self.dataSize = r.int64(f)

class EdgeBlock(DataBlock):
    def __init__(self,f):
        super().__init__(f)
        self.streamPath = ""
        self.EdgeDataOffset = 0
        self.posOffset = 0
        self.EdgeDataSize = 0
        self.posSize = 0

        self.ParseEdgeBlock(f)

    def ParseEdgeBlock(self,f):
        r = ByteReader()
        f.seek(36,1)
        pathLength = r.int32(f)
        self.streamPath = r.path(f,pathLength)
        self.EdgeDataOffset = r.int64(f)
        self.EdgeDataSize = r.int64(f)
        self.EndBlock(f)
class StuffBlock(DataBlock):
    def __init__(self,f):
        super().__init__(f)
        self.faceIndexCount = 0
        self.posOffset = 0

        self.ParseBlock(f)

    def ParseBlock(self,f):
        r = ByteReader()
        f.seek(87,1)
        self.posOffset = f.tell()
        self.faceIndexCount = r.int64(f)
        self.EndBlock(f)
class VertexBlock(DataBlock):
    def __init__(self,f):
        super().__init__(f,13522917709279820436)
        self.vertexCount = 0

        self.streamRefCount = 0
        self.inStream = True
        self.vertexStream = None
        self.normalsStream = None
        self.uvStream = None
        self.realOffsets = False
        self.hasVertexColor = False
        self.hasTwoUV = False
        self.coHalf = False #vertex coordinate stored as 16bit float

        #The positions at which the values are found
        self.posVCount = 0


        self.ParseVertexBlock(f)
        if self.inStream:
            self.CheckCoHalf()
        self.EndBlock(f)

    def ParseVertexBlock(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.posVCount = f.tell()
        self.vertexCount = r.int32(f)
        self.streamRefCount = r.int32(f)
        self.inStream = r.bool(f)
        # print(self.vertexCount,self.streamRefCount,self.inStream)
        if self.inStream:
            self.vertexStream = StreamRef(f,self.blockStartOffset,self.size)

            if self.streamRefCount == 3:
                self.normalsStream = StreamRef(f,self.blockStartOffset,self.size)
                self.uvStream = StreamRef(f,self.blockStartOffset,self.size)
            else:
                self.uvStream = StreamRef(f,self.blockStartOffset,self.size)

        if self.inStream:
            if self.normalsStream:
                self.realOffsets = self.vertexStream.dataOffset != self.normalsStream.dataOffset
            else:
                self.realOffsets = self.vertexStream.dataOffset != self.uvStream.dataOffset
            self.hasVertexColor = self.vertexCount * 8 < self.uvStream.dataSize
            self.hasTwoUV =self.vertexCount * 12 <= self.uvStream.dataSize


    def CheckCoHalf(self):
        vOffset = self.vertexStream.dataOffset
        HZDEditor = bpy.context.scene.HZDEditor
        core = HZDEditor.HZDAbsPath
        stream = core+".stream"

        s = open(stream,'rb')
        s.seek(vOffset)
        s.seek(6,1)
        if s.read(2) == b'\x00\x3C':
            self.coHalf = True
        else:
            self.coHalf = False
        s.close()
class FaceBlock(DataBlock):
    def __init__(self,f):
        super().__init__(f,12198706699739407665)
        self.indexCount = 0
        self.posIndexCount = 0
        self.inStream = True
        self.pathLength = 0
        self.streamPath = ""
        self.faceDataOffset = 0
        self.faceDataSize = 0
        self.posOffset = 0
        self.posSize = 0

        self.ParseFaceBlock(f)

    def ParseFaceBlock(self,f):
        r = ByteReader()
        f.seek(16, 1)
        self.posIndexCount = f.tell()
        self.indexCount = r.int32(f)
        f.seek(8,1)
        self.inStream = r.bool(f)
        f.seek(3,1)
        f.seek(16,1)
        if self.inStream:
            self.pathLength = r.int32(f)
            if 12 >= self.pathLength > 2000:  # kinda arbitrary numbers, was easier to put 2000 than getting file size.
                f.seek(self.blockStartOffset + self.size)
            else:
                self.streamPath = r.path(f, self.pathLength)
                self.posOffset = f.tell()
                self.faceDataOffset = r.int64(f)
                self.posSize = f.tell()
                self.faceDataSize = r.int64(f)
        self.EndBlock(f)

class MeshDataBlock:
    def __init__(self,f):
        self.edgeBlock = None
        self.stuffBlock = None
        self.vertexBlock = None
        self.faceBlock = None

        self.ReadMeshBlock(f)

    def ReadMeshBlock(self,f):
        r = ByteReader()
        # print(f.tell())
        IDCheck = r.int64(f)
        # print(f.tell())
        f.seek(-8,1)
        # print(f.tell())
        if IDCheck == 10234768860597628846:
            self.edgeBlock = EdgeBlock(f)
        self.stuffBlock = StuffBlock(f)
        self.vertexBlock = VertexBlock(f)
        self.faceBlock = FaceBlock(f)

class TextureRef():
    def __init__(self,f):
        self.texPath = ""

        r = ByteReader()
        f.seek(16,1)
        indicator = r.int8(f)
        f.seek(16,1)
        if indicator == 2:
            self.texPath = r.hashtext(f)
        f.seek(16,1)

class ShaderBlockInfo():
    def __init__(self,f):
        self.textureRefs = []

        self.ParseShaderInfo(f)

    def ParseShaderInfo(self,f):
        r = ByteReader()
        f.seek(19,1)
        bCount = r.int32(f)
        for b in range(bCount):
            f.seek(2,1)
        cCount = r.int32(f)
        for c in range(cCount):
            f.seek(8,1)
        f.seek(23,1)
        texRefCount = r.int32(f)
        for tex in range(texRefCount):
            self.textureRefs.append(TextureRef(f))
        gCount = r.int32(f)
        for g in range(gCount):
            f.seek(9,1)
            f.seek(16,1)
            f.seek(8,1)
        f.seek(25,1)

class MaterialBlock(DataBlock):
    def __init__(self,f):
        super().__init__(f,12029122079492233037)
        self.shaderName = ""
        self.subshaderCount = 0
        self.shaderBlockInfos = []
        self.ui_ShowTextures = False
        self.uniqueTextures = []

        self.ParseMaterialBlock(f)
        self.GetUniqueTexturesOfMatIndex()

    def ParseMaterialBlock(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.shaderName = r.hashtext(f)
        f.seek(4,1)
        self.subshaderCount = r.int32(f)
        for s in range(self.subshaderCount):
            self.shaderBlockInfos.append(ShaderBlockInfo(f))

        self.EndBlock(f)

    def GetUniqueTexturesOfMatIndex(self):
        for info in self.shaderBlockInfos:
            for tr in info.textureRefs:
                tp = tr.texPath
                if tp != "":
                    if tp not in self.uniqueTextures:
                        self.uniqueTextures.append(tp)
        return self.uniqueTextures

class LOD:
    def __init__(self,f):
        self.meshNameBlock = MeshNameBlock(f)
        self.skeletonBlock = SkeletonBlock(f)
        self.unknownBlock = DataBlock(f)
        self.unknownBlock.EndBlock(f)
        self.meshBlockInfo = MeshBlockInfo(f)
        self.meshBlockList = []
        self.materialBlockList = []


        for i in range(self.meshBlockInfo.meshBlockCount):
            self.meshBlockList.append(MeshDataBlock(f))
        for i in range(self.meshBlockInfo.meshBlockCount):
            material = MaterialBlock(f)
            c = 0
            while c < material.subshaderCount:
                db = DataBlock(f)
                if db.ID == 17501462827539052646: #Color Ramp
                    pass
                else:
                    c = c+1
                db.EndBlock(f)
            self.materialBlockList.append(material)


class LODGroup(DataBlock):
    def __init__(self, f):
        super().__init__(f)
        self.objectName = ""
        self.totalMeshCount = 0
        self.LODCount = 0
        self.LODList = []
        self.LODDistanceList = []

        self.ParseLODGroupInfo(f)

    def ParseLODGroupInfo(self,f):
        r = ByteReader()
        HZDEditor = bpy.context.scene.HZDEditor
        f.seek(16,1)
        self.objectName = r.hashtext(f)
        f.seek(24,1)
        f.seek(8,1)
        self.totalMeshCount = r.int32(f)
        f.seek(16,1)
        self.LODCount = r.int32(f)
        for i in range(self.LODCount):
            f.seek(17,1)
            lodDistance = r.float(f)
            HZDEditor["LodDistance" + str(i)] = lodDistance
            self.LODDistanceList.append(lodDistance)

        self.EndBlock(f)
        for i in range(self.LODCount- len(asset.LODObjects)):
            lod = LOD(f)
            self.LODList.append(lod)
class LODObject(DataBlock):
    def __init__(self, f):
        super().__init__(f)
        self.objectName = ""
        self.totalMeshCount = 0
        self.LODCount = 0
        self.LODList = []

        self.ParseLODObjectInfo(f)

    def ParseLODObjectInfo(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.objectName = r.hashtext(f)
        f.seek(24,1)
        f.seek(8,1)
        self.totalMeshCount = r.int32(f)
        f.seek(12,1)
        self.LODCount = r.int32(f)
        self.EndBlock(f)
        print(self.objectName,self.totalMeshCount,self.LODCount)

        for i in range(self.LODCount):
            lod = LOD(f)
            self.LODList.append(lod)
            # print("LOD", i,self.LODCount)


def ReadCoreFile():
    r = ByteReader()
    HZDEditor = bpy.context.scene.HZDEditor
    core = HZDEditor.HZDAbsPath
    coresize = os.path.getsize(core)

    global asset
    asset = Asset()

    with open(core, "rb") as f:
        while f.tell() < coresize:
            # print(f.tell(),coresize)
            ID = r.int64(f)
            f.seek(-8,1)
            # print(ID)
            if ID == 6871768592993170868: # LOD Group Info
                asset.LODGroups.append(LODGroup(f))
            elif ID == 7022335006738406101: # LOD Object Info
                asset.LODObjects.append(LODObject(f))

class SearchForOffsets(bpy.types.Operator):
    """Searches the .core file for offsets and sizes"""
    bl_idname = "object.hzd_offsets"
    bl_label = "Search Data"

    def execute(self,context):
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
        return {'FINISHED'}

class ImportHZD(bpy.types.Operator):
    """Imports the mesh"""
    bl_idname = "object.import_hzd"
    bl_label = ""

    isGroup: bpy.props.BoolProperty()
    Index: bpy.props.IntProperty()
    LODIndex: bpy.props.IntProperty()
    BlockIndex: bpy.props.IntProperty()


    def execute(self, context):
        ImportMesh(self.isGroup,self.Index,self.LODIndex,self.BlockIndex)
        return {'FINISHED'}
class ImportLodHZD(bpy.types.Operator):
    """Imports every mesh in the LOD"""
    bl_idname = "object.import_lod_hzd"
    bl_label = "Import"

    isGroup: bpy.props.BoolProperty()
    Index: bpy.props.IntProperty()
    LODIndex: bpy.props.IntProperty()


    def execute(self, context):
        if self.isGroup:
            for blockindex,block in enumerate(asset.LODGroups[self.Index].LODList[self.LODIndex].meshBlockList):
                ImportMesh(self.isGroup,self.Index,self.LODIndex,blockindex)
        else:
            for blockindex,block in enumerate(asset.LODObjects[self.Index].LODList[self.LODIndex].meshBlockList):
                ImportMesh(self.isGroup,self.Index,self.LODIndex,blockindex)
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

    isGroup: bpy.props.BoolProperty()
    Index: bpy.props.IntProperty()
    LODIndex: bpy.props.IntProperty()
    BlockIndex: bpy.props.IntProperty()

    def execute(self, context):
        ExportMesh(self.isGroup,self.Index,self.LODIndex,self.BlockIndex)
        return {'FINISHED'}
class ExportLodHZD(bpy.types.Operator):
    """Exports every mesh in the LOD"""
    bl_idname = "object.export_lod_hzd"
    bl_label = "Export"

    isGroup: bpy.props.BoolProperty()
    Index: bpy.props.IntProperty()
    LODIndex: bpy.props.IntProperty()


    def execute(self, context):
        if self.isGroup:
            for blockindex,block in enumerate(asset.LODGroups[self.Index].LODList[self.LODIndex].meshBlockList):
                ExportMesh(self.isGroup,self.Index,self.LODIndex,blockindex)
                ReadCoreFile()

        else:
            for blockindex,block in enumerate(asset.LODObjects[self.Index].LODList[self.LODIndex].meshBlockList):
                ExportMesh(self.isGroup,self.Index,self.LODIndex,blockindex)
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

    isGroup: bpy.props.BoolProperty()
    Index: bpy.props.IntProperty()
    LODIndex: bpy.props.IntProperty()
    BlockIndex: bpy.props.IntProperty()

    def execute(self,context):
        if self.isGroup:
            mat = asset.LODGroups[self.Index].LODList[self.LODIndex].materialBlockList[self.BlockIndex]
            mat.ui_ShowTextures = not mat.ui_ShowTextures
        else:
            mat = asset.LODObjects[self.Index].LODList[self.LODIndex].materialBlockList[self.BlockIndex]
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
        row.operator("object.hzd_offsets", icon='ZOOM_ALL')
        if BoneMatrices:
            row = layout.row()
            row.operator("object.import_skt",icon="ARMATURE_DATA")
            row = layout.row()
            row.prop(HZDEditor,"ExtractTextures")
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
            for ig, lg in enumerate(asset.LODGroups):
                box = mainRow.box()
                box.label(text="LOD DISTANCES", icon='OPTIONS')
                saveDistances = box.operator("object.savedistances")
                saveDistances.Index = ig
                for il, l in enumerate(lg.LODList):
                    lodBox = box.box()
                    disRow = lodBox.row()
                    disRow.prop(HZDEditor, "LodDistance" + str(il))

class LodObjectPanel(bpy.types.Panel):
    bl_label = "LOD Objects"
    bl_idname = "OBJECT_PT_lodobject"
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
            for io, lo in enumerate(asset.LODObjects):
                box = mainRow.box()
                box.label(text="LOD OBJECT", icon='SNAP_VOLUME')
                for il, l in enumerate(lo.LODList):
                    lodBox = box.box()
                    lodRow = lodBox.row()
                    lodRow.label(text="ELEMENT", icon='MATERIAL_DATA')
                    LODImport = lodRow.operator("object.import_lod_hzd", icon='IMPORT')
                    LODImport.isGroup = False
                    LODImport.Index = io
                    LODImport.LODIndex = il
                    LODExport = lodRow.operator("object.export_lod_hzd", icon='EXPORT')
                    LODExport.isGroup = False
                    LODExport.Index = io
                    LODExport.LODIndex = il
                    for ib, m in enumerate(l.meshBlockList):
                        row = lodBox.row()
                        row.label(text=str(ib) + "_" + l.meshNameBlock.name + " " + str(m.vertexBlock.vertexCount),
                                  icon='MESH_ICOSPHERE')
                        if m.vertexBlock.inStream:
                            if l.materialBlockList[ib].ui_ShowTextures:
                                texIcon = 'UV'
                            else:
                                texIcon = 'TEXTURE'

                            texButton = row.operator("object.usedtextures", icon=texIcon)
                            texButton.isGroup = True
                            texButton.Index = io
                            texButton.LODIndex = il
                            texButton.BlockIndex = ib

                            Button = row.operator("object.import_hzd", icon='IMPORT')
                            Button.isGroup = False
                            Button.Index = io
                            Button.LODIndex = il
                            Button.BlockIndex = ib
                            Button = row.operator("object.export_hzd", icon='EXPORT')
                            Button.isGroup = False
                            Button.Index = io
                            Button.LODIndex = il
                            Button.BlockIndex = ib

                            if l.materialBlockList[ib].ui_ShowTextures:
                                texBox = lodBox.box()

                                for t in l.materialBlockList[ib].uniqueTextures:
                                    texRow = texBox.row()
                                    texRow.label(text=t)

                        else:
                            row.label(text="Not able to Import for now.")

class LodGroupPanel(bpy.types.Panel):
    bl_label = "LOD Groups"
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
            for ig, lg in enumerate(asset.LODGroups):
                box = mainRow.box()
                box.label(text="LOD GROUP", icon='STICKY_UVS_LOC')

                for il, l in enumerate(lg.LODList):
                    lodBox = box.box()
                    lodRow = lodBox.row()
                    lodRow.label(text="LOD", icon='MOD_EXPLODE')

                    LODImport = lodRow.operator("object.import_lod_hzd", icon='IMPORT')
                    LODImport.isGroup = True
                    LODImport.Index = ig
                    LODImport.LODIndex = il
                    LODExport = lodRow.operator("object.export_lod_hzd", icon='EXPORT')
                    LODExport.isGroup = True
                    LODExport.Index = ig
                    LODExport.LODIndex = il

                    for ib, m in enumerate(l.meshBlockList):
                        row = lodBox.row()
                        row.label(text=str(ib) + "_" + l.meshNameBlock.name + " " + str(m.vertexBlock.vertexCount),
                                  icon='MESH_ICOSPHERE')
                        if m.vertexBlock.inStream:
                            if l.materialBlockList[ib].ui_ShowTextures:
                                texIcon = 'UV'
                            else:
                                texIcon = 'TEXTURE'

                            texButton = row.operator("object.usedtextures", icon=texIcon)
                            texButton.isGroup = True
                            texButton.Index = ig
                            texButton.LODIndex = il
                            texButton.BlockIndex = ib

                            Button = row.operator("object.import_hzd", icon='IMPORT')
                            Button.isGroup = True
                            Button.Index = ig
                            Button.LODIndex = il
                            Button.BlockIndex = ib
                            Button = row.operator("object.export_hzd", icon='EXPORT')
                            Button.isGroup = True
                            Button.Index = ig
                            Button.LODIndex = il
                            Button.BlockIndex = ib

                            if l.materialBlockList[ib].ui_ShowTextures:
                                texBox = lodBox.box()

                                for t in l.materialBlockList[ib].uniqueTextures:
                                    texRow = texBox.row()
                                    texRow.label(text=t)

                        else:
                            row.label(text="Not able to Import for now.")

classes=[ImportHZD,
         ImportLodHZD,
         ImportSkeleton,
         ExportHZD,
         ExportLodHZD,
         SaveLodDistances,
         HZDSettings,
         SearchForOffsets,
         HZDPanel,
         LODDistancePanel,
         LodObjectPanel,
         LodGroupPanel,
         ShowUsedTextures,
         ExtractHZDAsset]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.HZDEditor = bpy.props.PointerProperty(type=HZDSettings)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
if __name__ == "__main__":
    register()
