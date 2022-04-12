# HZDMeshTool
Import/Export meshes from Horizon Zero Dawn's .core format

## Download
https://github.com/AlexP0/HZDMeshTool/releases

## Installation
- In Blender open the User Preferences
- Go in the Add-ons tab and click Install...
- Select the HZDMeshTool.zip file
- Enable the addon

## Usage
### 1 - Go to Scene Properties tab

### 2 - As of 1.3 you have two ways of importing meshes
### **2A** - **Extract the files directly from the game.**
- - **A-1 Set the Workspace Path**
>This is the path where assets will be extracted from the game (in a file tree)
- - **A-2 Set the Game Path** 
>This is the path to the **folder** where HorizonZeroDawn.exe is located
- - **A-3 Set the Asset Path** 
>This is the archive path to a skeletal mesh (e.g.,models/characters/humans/aloy/animation/parts/aloy_ct_a_lx.core)
- - **A-4 Click Extract** 
>This will extract the mesh .core and .stream along with the appropriate skeleton .core **directly** from the game files. 
It will also auto-input the paths in the Mesh Core and Skeleton Core fields and will auto-click Search Data so you're ready for import. 
(In general the tool will never extract over existing files, delete the files manually if you want to extract them again)

### **2B** - **Use already extract files (from ProjectDecima or other extraction tool)**
- - **Input a file in Mesh Core and Skeleton Core.**
>Must be .core files, the associated .stream must be in the same directory and have the same name. 
    The skeleton must be the one referenced in by the mesh.

- - **Click Search Data**
>The tool will search for and display a list of mesh parts found in the .core file.
    Meshes are part of either a LOD Object (contains 1 LOD with multiple elements) or a LOD Group (contains multiple LODs)
    Lowest level LODs aren't handled yet because they are stored directly in the .core
    Mesh parts with a warning sign are not safe to export yet.

### 3 - Click Import icon next to one of the mesh parts in the LOD Objects or LOD Groups sub-panels
>A mesh will be created and placed in organized collection. The tool will search for existing skeleton based on the armature data number, it will import one from the skeleton .core if not found. You can also click the button named "Import" to import the entire LOD. Take note that some mesh have multiple UVs or Vertex Color channels.

#### 3A - Texture Extract and Material Creation
>If you setup Game Path and Workspace Path in set 2A, you can enable the Extract Texture checkbox. If enabled, the tool will extract every textures used by the imported mesh part to the Workspace directory. You can click the checkered icon on next to the mesh import button to see the textures used.
>
>Extracting textures will also cause the tool to create a material (if it doesn't already exist) and place the textures in it.
>Texture Sets will be built into Node Groups with outputs coresponding to the in-game usage of the texture or it's RGBA channels.
>No connection will be made to the shader node, I have no way of knowing which texture goes where yet.
    

### 4 - Do your edits  
>You can edit the mesh however you want, but:
    -You must keep the Vertex Group order intact
    -You must have the same amount of UVs/vColor channels as was imported
    -Modifications to the skeleton are not considered
    
### 5 - Click Export Button (to export an entire lod) or the export icon for a singular mesh part.
    
>The original files will be overwritten with the new data.
    The tool will export the Object that has the same name as the mesh part (e.i.: "0_Eyelashes")
    The number after the name is the vertex count of the original mesh, it is not relevant to export. 
    (only to give an idea of the size of the mesh data on import)
    Again: Mesh parts with a warning sign are not safe to export yet.
    
### 6 - Modifying LOD Distances.

>You may also modify the distances (in meters) at which the lod will appear in game.
    Click Save LOD Distances button to save them on the file.
