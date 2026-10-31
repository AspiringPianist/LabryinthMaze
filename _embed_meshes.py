import json
import pathlib

keep = ["airplane", "teapot", "shark", "ant", "scissors"]
root = pathlib.Path("meshes")
data = {}
for name in keep:
    text = (root / f"{name}.ply").read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    index = 0
    vertex_count = 0
    face_count = 0
    while lines[index] != "end_header":
        if lines[index].startswith("element vertex"):
            vertex_count = int(lines[index].split()[-1])
        if lines[index].startswith("element face"):
            face_count = int(lines[index].split()[-1])
        index += 1
    index += 1
    faces = lines[index + vertex_count : index + vertex_count + face_count]
    if len(faces) != face_count:
        raise SystemExit(f"{name} face count mismatch")
    for face in faces:
        parts = face.split()
        count = int(parts[0])
        indices = [int(part) for part in parts[1 : 1 + count]]
        if any(item < 0 or item >= vertex_count for item in indices):
            raise SystemExit(f"{name} bad index")
    data[name] = text
    print(name, "verts", vertex_count, "faces", face_count)

header = (
    "/* ASCII PLY text from John Burkardt's public PLY archive. "
    "parsePLY() in main.js reads these at startup. */\n"
)
pathlib.Path("meshes.js").write_text(
    header + "const PLY_TEXT = " + json.dumps(data) + ";\n",
    encoding="utf-8",
)
print("wrote meshes.js", pathlib.Path("meshes.js").stat().st_size)

for path in root.glob("*.ply"):
    if path.stem not in keep:
        path.unlink()
        print("removed", path.name)
