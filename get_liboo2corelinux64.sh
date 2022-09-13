#!/usr/bin/bash

if [[ $# -gt 0 ]]; then
    GitDepsXml=$(<"$1")
else
    cat <<-EOF
	usage: $0 Commit.gitdeps.xml
	
	To extract the linux oddle library liboo2corelinux64.so.9 from UnrealEngine,
	you need first follow the instruction at
	https://github.com/EpicGames/Signup
	to get access to the UnrealEngine github repo.
	Then you need to download
	https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Build/Commit.gitdeps.xml
	and call $0 with path to Commit.gitdeps.xml.
	EOF
fi
BaseUrl=$(echo "$GitDepsXml" | sed -n -e "/BaseUrl/ s/.*BaseUrl=\"\([^\"]*\)\".*/\1/g p;")
Files=$(echo "$GitDepsXml" | sed -n -e "/File Name=\"[^\"]*liboo2corelinux64.so[^\"]*\"/ s/.*File Name=\"\([^\"]*\)\".*/\1/g p;")
for FileName in $Files; do
    OutFile=${FileName##*/}
    Hash=$(echo "$GitDepsXml" | sed -n -e "/File Name=\"${FileName////.}\"/ s/.*Hash=\"\([^\"]*\)\".*/\1/g p;")
    PackHash=$(echo "$GitDepsXml" | sed -n -e "/Blob Hash=\"$Hash\"/ s/.*PackHash=\"\([^\"]*\)\".*/\1/g p;")
    PackOffset=$(echo "$GitDepsXml" | sed -n -e "/Blob Hash=\"$Hash\"/ s/.*PackOffset=\"\([^\"]*\)\".*/\1/g p;")
    Size=$(echo "$GitDepsXml" | sed -n -e "/Blob Hash=\"$Hash\"/ s/.*Size=\"\([^\"]*\)\".*/\1/g p;")
    RemotePath=$(echo "$GitDepsXml" | sed -n -e "/Pack Hash=\"$PackHash\"/ s/.*RemotePath=\"\([^\"]*\)\".*/\1/g p;")
    echo "$BaseUrl/$RemotePath/$PackHash -> $OutFile [$PackOffset:+$Size]"
    curl -s $BaseUrl/$RemotePath/$PackHash | zcat | dd if=/dev/stdin of=$OutFile bs=1 skip=$PackOffset count=$Size
done
