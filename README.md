# HZDMeshTool
Import/Export meshes from Horizon Zero Dawn's .core format

### Download
https://github.com/AlexP0/HZDMeshTool/releases/tag/1.0.4

### Installation
- In Blender open the User Preferences
- Go in the Add-ons tab and click Install...
- Select the HZDMeshTool.zip file
- Enable the addon

### Usage
1- Go to Scene Properties tab

2- Input a file in Mesh Core and Skeleton Core.
    
    Must be .core files, the associated .stream must be in the same directory and have the same name. 
    The skeleton must be the one referenced in by the mesh.

3- Click Search Data

    The tool will search for and display a list of mesh parts found in the .core file.
    Mesh parts usually represent Material IDs and LODs. 
    Lowest level LODs aren't handled yet because they are stored directly in the .core
    Mesh parts with a warning sign are not safe to export yet.

4- Click Import Skeleton
    
    The skeleton needs to exist in blender and be assigned to the script for the mesh to import
    The name of the skeleton doesn't really matter, only that the script knows about it.

5- Click Import button next to one of the mesh parts.
    
    A mesh will be created and assigned to the Skeleton.
    Take note that some mesh have multiple UVs or Vertex Color channels.

6- Do your edits
    
    You can edit the mesh however you want, but:
    -You must keep the Vertex Group order intact
    -You must have the same amount of UVs/vColor channels as was imported
    -Modifications to the skeleton are not considered
    
7- Click Export Button
    
    The tool will export the Object that has the same name as the mesh part (e.i.: "0_Eyelashes")
    The number after the name is the vertex count of the original mesh, it is not relevant to export. 
    (only to give an idea of the size of the mesh data on import)
    Again: Mesh parts with a warning sign are not safe to export yet.
    
8- .coreMOD and .streamMOD files get created in the same location as the original file.

    These are the files containing your edits. When repacking for HZD simply rename the file's extension by removing "MOD".
    You may export multiple times over the same MOD file (for multiple mesh parts edits).
    The tool will detect existing .coreMOD file in the directoy and update the MOD file 
    instead of creating one from the original file.
