bl_info = {
    "name": "HZD Mesh Tool",
    "author": "AlexPo",
    "location": "Scene Properties > HZD Panel",
    "version": (1, 2, 0),
    "blender": (2, 93, 0),
    "description": "This addon imports/exports skeletal meshes\n from Horizon Zero Dawn's .core/.stream files",
    "category": "Import-Export"
    }

import bpy
import bmesh
import os
import pathlib
from struct import unpack, pack
import math
import numpy as np
import mathutils
import operator


BoneMatrices = {}



def ClearProperties(self,context):
    HZDEditor = bpy.context.scene.HZDEditor
    HZDEditor.HZDAbsPath = bpy.path.abspath(HZDEditor.HZDPath)
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
    def int32(f):
        b = f.read(4)
        i = unpack('<i',b)[0]
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
        f.seek(int(bic - len(boneIndices)),1)
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
    return [nx,ny,nz], [tx,ty,tz], flip
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
    HZDPath: bpy.props.StringProperty(name="Mesh Core",subtype='FILE_PATH', update=ClearProperties)
    HZDAbsPath : bpy.props.StringProperty()
    HZDSize: bpy.props.IntProperty()
    SkeletonPath: bpy.props.StringProperty(name="Skeleton Core",subtype='FILE_PATH', update=ClearProperties)
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


def ImportMesh(isGroup,Index,LODIndex,BlockIndex):
    r = ByteReader()
    print(isGroup,Index,LODIndex,BlockIndex)
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
    boneCount = len(bpy.data.objects[HZDEditor.SkeletonName].data.bones)

    print(meshName,"Real Offsets = ",md.vertexBlock.realOffsets, "vOffset = ",md.vertexBlock.vertexStream.dataOffset, "vStride = ", md.vertexBlock.vertexStream.stride)

    print(md.vertexBlock.vertexCount)

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
        for n in range(fb.faceDataOffset,fb.faceDataOffset+fb.faceDataSize,6):
            face = ParseFaces(f)
            if face[0] < vertCount and face[1] < vertCount and face[2] < vertCount:
                fList.append(face)

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
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # VERTICES ####################################
    for v in vList:
        vert = bm.verts.new()
        vert.co = v
    bm.verts.ensure_lookup_table()
    # TRIANGLES ####################################
    for f in fList:
        if f[0] != f[1] != f[2]:
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

    # NORMALS #######################################

    if vb.normalsStream:
        for i,v in enumerate(mesh.vertices):
            v.normal = nList[i]

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

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
    armature = bpy.data.objects[HZDEditor.SkeletonName].data
    for v in mesh.vertices:
        vindex = v.index
        # print(vindex, biList[vindex],bwList[vindex])
        for index, boneindex in enumerate(biList[vindex]):
            # print(len(biList[vindex])-1- index, "=", len(biList[vindex]), "-", index)
            index = index - 1
            if index == -1:
                index = len(biList[vindex]) - 1
                # print(vindex, index,bwList[vindex][index])
            # print(len(CoreBones)-1,boneindex,"   ",len(bwList)-1,vindex)
            obj.vertex_groups[CoreBones[boneindex]].add([vindex], bwList[vindex][index], "ADD")


    bpy.context.collection.objects.link(obj)

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
            Bones.append(boneName)
            ParentIndices.append(parentIndex)

    armature = bpy.data.armatures.new(sktName+"Data")
    obj = bpy.data.objects.new(sktName, armature)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    HZDEditor.SkeletonName = obj.name

    bpy.ops.object.mode_set(mode="EDIT")

    for i,b in enumerate(Bones):
        bone = armature.edit_bones.new(b)
        bone.parent = armature.edit_bones[ParentIndices[i]]
        bone.tail = mathutils.Vector([0,0,1])

    for b in BoneMatrices:
        bone = armature.edit_bones[b]
        bone.tail = mathutils.Vector([0,0,1])
        bone.transform(BoneMatrices[b])

    bpy.ops.object.mode_set(mode='OBJECT')

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
        bic = int((stride - coLength) / 3)
        bint16 = True
    else:
        bic = int((stride - coLength) / 2)  # bone indices count
        bint16 = False
    # Gather bone groups
    groupsweights = {}
    for vg in vertex.groups:
        if vg.weight > 0.0:
            groupsweights[vg.group] = vg.weight
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
    for b in range(len(bVertex),vbLength):
        bVertex += b'\x00'
    # print(bic, boneRepeat,len(bVertex))
    if len(bVertex) == vbLength:
        f.write(bVertex)
    else:
        raise Exception("Vertex bytes not expected length:{v} instead of {e}".format(v=len(bVertex),e=vbLength))
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
    def __init__(self,f):
        r = ByteReader()
        self.ID = r.int64(f)
        self.size = r.int32(f)
        self.blockStartOffset = f.tell()
        # print(self.__class__.__name__)
        # print("ID = ",self.ID,"\n","Size = ",self.size,"\nStart = ",self.blockStartOffset)
    def EndBlock(self,f):
        f.seek(self.blockStartOffset + self.size)

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
        super().__init__(f)
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
        super().__init__(f)
        self.name = ""

        self.ParseMeshName(f)

    def ParseMeshName(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.name = r.hashtext(f)
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
        super().__init__(f)
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
        super().__init__(f)
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
        super().__init__(f)
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

class ShaderBlock(DataBlock):
    def __init__(self,f):
        super().__init__(f)
        self.shaderName = ""
        self.subshaderCount = 0

        self.ParseShaderBlock(f)

    def ParseShaderBlock(self,f):
        r = ByteReader()
        f.seek(16,1)
        self.shaderName = r.hashtext(f)
        f.seek(4,1)
        self.subshaderCount = r.int32(f)
        self.EndBlock(f)

class LOD:
    def __init__(self,f):
        self.meshNameBlock = MeshNameBlock(f)
        self.skeletonBlock = SkeletonBlock(f)
        self.unknownBlock = DataBlock(f)
        self.unknownBlock.EndBlock(f)
        self.meshBlockInfo = MeshBlockInfo(f)
        self.meshBlockList = []
        self.shaderBlockList = []


        for i in range(self.meshBlockInfo.meshBlockCount):
            self.meshBlockList.append(MeshDataBlock(f))
        for i in range(self.meshBlockInfo.meshBlockCount):
            shader = ShaderBlock(f)
            c = 0
            while c < shader.subshaderCount:
                db = DataBlock(f)
                if db.ID == 17501462827539052646: #Color Ramp
                    pass
                else:
                    c = c+1
                db.EndBlock(f)
            self.shaderBlockList.append(shader)
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
        CreateSkeleton()
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
        row.prop(HZDEditor,"HZDPath")
        row = layout.row()
        row.prop(HZDEditor, "SkeletonPath")
        row = layout.row()
        row.label(text=HZDEditor.SkeletonName)

        row = layout.row()
        row.operator("object.hzd_offsets", icon='ZOOM_ALL')
        if BoneMatrices:
            row = layout.row()
            row.operator("object.import_skt",icon="ARMATURE_DATA")
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

                        else:
                            row.label(text="Not able to Import for now.")

def register():
    bpy.utils.register_class(ImportHZD)
    bpy.utils.register_class(ImportLodHZD)
    bpy.utils.register_class(ImportSkeleton)
    bpy.utils.register_class(ExportHZD)
    bpy.utils.register_class(ExportLodHZD)
    bpy.utils.register_class(SaveLodDistances)
    bpy.utils.register_class(HZDSettings)
    bpy.utils.register_class(SearchForOffsets)
    bpy.types.Scene.HZDEditor = bpy.props.PointerProperty(type=HZDSettings)
    bpy.utils.register_class(HZDPanel)
    bpy.utils.register_class(LODDistancePanel)
    bpy.utils.register_class(LodObjectPanel)
    bpy.utils.register_class(LodGroupPanel)
def unregister():
    bpy.utils.unregister_class(HZDPanel)
    bpy.utils.unregister_class(LODDistancePanel)
    bpy.utils.unregister_class(LodObjectPanel)
    bpy.utils.unregister_class(LodGroupPanel)
    bpy.utils.unregister_class(ImportHZD)
    bpy.utils.unregister_class(ImportLodHZD)
    bpy.utils.unregister_class(ImportSkeleton)
    bpy.utils.unregister_class(ExportHZD)
    bpy.utils.unregister_class(ExportLodHZD)
    bpy.utils.unregister_class(SaveLodDistances)
    bpy.utils.unregister_class(SearchForOffsets)
if __name__ == "__main__":
    register()