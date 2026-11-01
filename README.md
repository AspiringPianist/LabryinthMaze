# Labyrinth Builder

WebGL labyrinth for CSE606. Open `index.html` in a browser. If local file loading is blocked:

```powershell
py -m http.server 8000
```

Then open http://localhost:8000.

## Obstacle models

The five obstacles are ASCII PLY meshes from John Burkardt's public archive. They are stored under `meshes/` and embedded in `meshes.js`.

- Archive index: https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html
- Airplane: https://people.sc.fsu.edu/~jburkardt/data/ply/airplane.ply
- Teapot: https://people.sc.fsu.edu/~jburkardt/data/ply/teapot.ply
- Shark: https://people.sc.fsu.edu/~jburkardt/data/ply/shark.ply
- Ant: https://people.sc.fsu.edu/~jburkardt/data/ply/ant.ply
- Scissors: https://people.sc.fsu.edu/~jburkardt/data/ply/scissors.ply

Walls, floors, the player marker, and the end pyramid are generated in `main.js`. They are not from that archive.

## API docs

```powershell
npx --yes jsdoc -c jsdoc.json
```

The generated pages are in `docs/jsdoc/index.html`.
