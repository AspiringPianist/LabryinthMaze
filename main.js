/**
 * Labyrinth builder for CSE606.
 *
 * The furniture planner (A1) got a canvas, compiled two shaders, linked a
 * program, uploaded a buffer, and drew with a model matrix. This file keeps
 * that order and extends it to 3D: a vec3 position, a normal, a view matrix,
 * and a projection matrix.
 *
 * World coordinates are Z-up. The floor is the XY plane, Z is the vertical
 * axis. That matches the assignment's player rotations: Z turns, Y looks up
 * and down, and X looks side to side.
 *
 * A model matrix is T * R * S. gl-matrix stores matrices in column-major
 * order, and the vertex shader multiplies on the left, so the rightmost
 * matrix hits the local point first:
 *
 *   clip = projection * view * translation * rotation * scale * localPoint
 *
 * Rotation is a quaternion. The trackball builds a quaternion from two
 * points on a sphere and multiplies it into the selected object.
 */

const { mat4, mat3, vec3, vec4, quat } = glMatrix;

const COLS = 8;
const ROWS = 8;
const WALL_H = 1.05;
const WALL_T = 0.08;
const PLAYER_RADIUS = 0.16;
const MAX_OBSTACLES = 5;
const MESH_ORDER = ["airplane", "teapot", "shark", "ant", "scissors"];
const MESH_COLOR = {
    airplane: [0.78, 0.24, 0.18, 1],
    teapot: [0.22, 0.42, 0.78, 1],
    shark: [0.12, 0.55, 0.62, 1],
    ant: [0.48, 0.3, 0.16, 1],
    scissors: [0.55, 0.22, 0.58, 1]
};

const canvas = document.getElementById("glCanvas");
const modeSelect = document.getElementById("modeSelect");
const selectionText = document.getElementById("selectionText");
const statusText = document.getElementById("statusText");
const modeHelp = document.getElementById("modeHelp");
const axesToggle = document.getElementById("axesToggle");
const gl = canvas.getContext("webgl");

/**
 * @typedef {Object} Obstacle
 * @property {number} id
 * @property {string} mesh Mesh name in PLY_TEXT.
 * @property {number} c Column on the maze grid.
 * @property {number} r Row on the maze grid.
 * @property {number} offsetX Visual shift inside the cell.
 * @property {number} offsetY Visual shift inside the cell.
 * @property {quat} quat Orientation quaternion.
 * @property {number} scale Uniform scale.
 * @property {("spin"|"patrol"|null)} live Idle motion, or null when the obstacle stays still.
 * @property {number} phase Phase offset for the patrol.
 */

/** @type {"top"|"world"|"play"} */
let mode = "top";
let axesVisible = true;
let time = 0;
let meshCursor = 0;
let nextId = 1;
let statusMessage = "";

/** @type {boolean[][]} Wall on y = r, covering x in [c, c + 1]. */
let hWall = [];
/** @type {boolean[][]} Wall on x = c, covering y in [r, r + 1]. */
let vWall = [];
/** @type {Obstacle[]} */
let obstacles = [];
/** @type {{c: number, r: number}} */
let end = { c: COLS - 1, r: ROWS - 1 };
const player = {
    x: 0.5,
    y: 0.5,
    yaw: 0,
    pitch: 0,
    roll: 0,
    quat: quat.create(),
    speed: 1.5
};
/** @type {{kind: string, id?: number}|null} */
let selection = null;
let pointer = null;
const held = new Set();
let floorBounds = { left: -0.5, right: COLS + 0.5, bottom: -0.5, top: ROWS + 0.5 };
let orbitYaw = 0.75;
let orbitPitch = 0.9;
let orbitDist = 12;

let positionLocation = -1;
let normalLocation = -1;
let modelLocation = null;
let viewLocation = null;
let projectionLocation = null;
let normalMatrixLocation = null;
let colorLocation = null;
let litLocation = null;
let fogLocation = null;
let fogColorLocation = null;
const crosshair = document.getElementById("crosshair");
let boxMesh = null;
let circleMesh = null;
let pyramidMesh = null;
/** @type {Object<string, {positionBuffer: WebGLBuffer, normalBuffer: WebGLBuffer, count: number}>} */
const library = {};

/**
 * Compiles one shader and throws with the driver log when compilation fails.
 * @param {WebGLRenderingContext} context
 * @param {number} type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
 * @param {string} source
 * @returns {WebGLShader}
 */
function createShader(context, type, source) {
    const shader = context.createShader(type);
    context.shaderSource(shader, source);
    context.compileShader(shader);
    if (!context.getShaderParameter(shader, context.COMPILE_STATUS)) {
        const log = context.getShaderInfoLog(shader);
        context.deleteShader(shader);
        throw new Error(log || "Shader compile failed");
    }
    return shader;
}

/**
 * Attaches the two shaders and links the program.
 * @param {WebGLRenderingContext} context
 * @param {WebGLShader} vertexShader
 * @param {WebGLShader} fragmentShader
 * @returns {WebGLProgram}
 */
function createProgram(context, vertexShader, fragmentShader) {
    const program = context.createProgram();
    context.attachShader(program, vertexShader);
    context.attachShader(program, fragmentShader);
    context.linkProgram(program);
    if (!context.getProgramParameter(program, context.LINK_STATUS)) {
        throw new Error(context.getProgramInfoLog(program) || "Program link failed");
    }
    return program;
}

/**
 * @param {string} text
 * @param {string} [kind]
 */
function setStatus(text, kind) {
    statusMessage = text;
    statusText.textContent = text;
    statusText.className = kind === "warn" ? "warn" : "";
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

/**
 * Column-major model matrix. Translation, then quaternion rotation, then scale,
 * which applies as scale, then rotation, then translation.
 * @param {number[]} translation
 * @param {quat|null} rotation
 * @param {number[]|null} scale
 * @returns {mat4}
 */
function trs(translation, rotation, scale) {
    const matrix = mat4.create();
    mat4.translate(matrix, matrix, translation);
    if (rotation) {
        const rotationMatrix = mat4.create();
        mat4.fromQuat(rotationMatrix, rotation);
        mat4.multiply(matrix, matrix, rotationMatrix);
    }
    if (scale) mat4.scale(matrix, matrix, scale);
    return matrix;
}

/**
 * @param {number[]} positions
 * @param {number[]} normals
 * @param {number[]} a
 * @param {number[]} b
 * @param {number[]} c
 * @param {number[]} normal
 */
function pushTri(positions, normals, a, b, c, normal) {
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    for (let i = 0; i < 3; i++) normals.push(normal[0], normal[1], normal[2]);
}

function faceNormal(a, b, c) {
    const ux = b[0] - a[0];
    const uy = b[1] - a[1];
    const uz = b[2] - a[2];
    const vx = c[0] - a[0];
    const vy = c[1] - a[1];
    const vz = c[2] - a[2];
    return [
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx
    ];
}

function createBox() {
    const positions = [];
    const normals = [];
    const faces = [
        [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0, 0, 1]],
        [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0, 0, -1]],
        [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0, 1, 0]],
        [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [0, -1, 0]],
        [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [1, 0, 0]],
        [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-1, 0, 0]]
    ];
    for (const face of faces) {
        const normal = face[4];
        pushTri(positions, normals, face[0], face[1], face[2], normal);
        pushTri(positions, normals, face[0], face[2], face[3], normal);
    }
    return { positions, normals };
}

function createCircle() {
    const positions = [];
    const normals = [];
    const segments = 28;
    for (let i = 0; i < segments; i++) {
        const a0 = (i / segments) * Math.PI * 2;
        const a1 = ((i + 1) / segments) * Math.PI * 2;
        pushTri(
            positions,
            normals,
            [0, 0, 0],
            [Math.cos(a0), Math.sin(a0), 0],
            [Math.cos(a1), Math.sin(a1), 0],
            [0, 0, 1]
        );
    }
    return { positions, normals };
}

function createPyramid() {
    const positions = [];
    const normals = [];
    const apex = [0, 0, 1];
    const base = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]];
    for (let i = 0; i < 4; i++) {
        const a = base[i];
        const b = base[(i + 1) % 4];
        const normal = faceNormal(a, b, apex);
        pushTri(positions, normals, a, b, apex, normal);
    }
    pushTri(positions, normals, base[0], base[2], base[1], [0, 0, -1]);
    pushTri(positions, normals, base[0], base[3], base[2], [0, 0, -1]);
    return { positions, normals };
}

/**
 * Reads an ASCII PLY file. File coordinates are Y-up, so they are swapped
 * into this project's Z-up world before the mesh is centered on the floor.
 * @param {string} text
 * @returns {{positions: number[], normals: number[]}}
 */
function parsePLY(text) {
    const lines = text.split(/\r?\n/);
    let index = 0;
    let vertexCount = 0;
    let faceCount = 0;
    while (index < lines.length && lines[index].trim() !== "end_header") {
        const line = lines[index].trim();
        if (line.startsWith("element vertex")) vertexCount = Number(line.split(/\s+/)[2]);
        if (line.startsWith("element face")) faceCount = Number(line.split(/\s+/)[2]);
        index++;
    }
    index++;
    const vertices = [];
    let read = 0;
    while (read < vertexCount && index < lines.length) {
        const line = lines[index++].trim();
        if (!line) continue;
        const parts = line.split(/\s+/).map(Number);
        vertices.push([parts[0], parts[2], parts[1]]);
        read++;
    }
    const faces = [];
    let facesRead = 0;
    while (facesRead < faceCount && index < lines.length) {
        const line = lines[index++].trim();
        if (!line) continue;
        const parts = line.split(/\s+/).map(Number);
        const count = parts[0];
        for (let k = 1; k < count - 1; k++) faces.push([parts[1], parts[1 + k], parts[2 + k]]);
        facesRead++;
    }
    return buildMesh(vertices, faces);
}

/**
 * Centers the mesh on X and Y, puts its base on Z = 0, and scales the largest
 * side to 1. Vertex normals are the average of the neighboring face normals.
 * @param {number[][]} vertices
 * @param {number[][]} faces
 * @returns {{positions: number[], normals: number[]}}
 */
function buildMesh(vertices, faces) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const vertex of vertices) {
        for (let k = 0; k < 3; k++) {
            min[k] = Math.min(min[k], vertex[k]);
            max[k] = Math.max(max[k], vertex[k]);
        }
    }
    const centerX = (min[0] + max[0]) / 2;
    const centerY = (min[1] + max[1]) / 2;
    const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
    for (const vertex of vertices) {
        vertex[0] = (vertex[0] - centerX) / size;
        vertex[1] = (vertex[1] - centerY) / size;
        vertex[2] = (vertex[2] - min[2]) / size;
    }
    const accumulated = vertices.map(() => [0, 0, 0]);
    const triangles = [];
    for (const [i, j, k] of faces) {
        const normal = faceNormal(vertices[i], vertices[j], vertices[k]);
        const length = Math.hypot(normal[0], normal[1], normal[2]);
        if (length < 1e-8) continue;
        normal[0] /= length;
        normal[1] /= length;
        normal[2] /= length;
        for (const vertexIndex of [i, j, k]) {
            accumulated[vertexIndex][0] += normal[0];
            accumulated[vertexIndex][1] += normal[1];
            accumulated[vertexIndex][2] += normal[2];
        }
        triangles.push(i, j, k);
    }
    for (const normal of accumulated) {
        const length = Math.hypot(normal[0], normal[1], normal[2]);
        if (length > 1e-8) {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            normal[2] = 1;
        }
    }
    const positions = [];
    const normals = [];
    for (let t = 0; t < triangles.length; t += 3) {
        for (const vertexIndex of [triangles[t], triangles[t + 1], triangles[t + 2]]) {
            positions.push(vertices[vertexIndex][0], vertices[vertexIndex][1], vertices[vertexIndex][2]);
            normals.push(accumulated[vertexIndex][0], accumulated[vertexIndex][1], accumulated[vertexIndex][2]);
        }
    }
    return { positions, normals };
}

function uploadMesh(geometry) {
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.positions), gl.STATIC_DRAW);
    const normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.normals), gl.STATIC_DRAW);
    return {
        positionBuffer,
        normalBuffer,
        count: geometry.positions.length / 3
    };
}

function drawMesh(mesh, matrix, color, lit) {
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.positionBuffer);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.normalBuffer);
    gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(modelLocation, false, matrix);
    const normalMatrix = mat3.create();
    if (!mat3.normalFromMat4(normalMatrix, matrix)) mat3.identity(normalMatrix);
    gl.uniformMatrix3fv(normalMatrixLocation, false, normalMatrix);
    gl.uniform4fv(colorLocation, color);
    gl.uniform1f(litLocation, lit);
    gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
}

function shuffle(list) {
    for (let i = list.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const swap = list[i];
        list[i] = list[j];
        list[j] = swap;
    }
    return list;
}

function generateMaze() {
    hWall = Array.from({ length: ROWS + 1 }, () => Array(COLS).fill(true));
    vWall = Array.from({ length: ROWS }, () => Array(COLS + 1).fill(true));
    const visited = Array.from({ length: ROWS }, () => Array(COLS).fill(false));
    const stack = [[0, 0]];
    visited[0][0] = true;
    const directions = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    while (stack.length) {
        const [c, r] = stack[stack.length - 1];
        const options = [];
        for (const [dc, dr] of directions) {
            const nc = c + dc;
            const nr = r + dr;
            if (nc >= 0 && nr >= 0 && nc < COLS && nr < ROWS && !visited[nr][nc]) {
                options.push([nc, nr, dc, dr]);
            }
        }
        if (!options.length) {
            stack.pop();
            continue;
        }
        const choice = options[Math.floor(Math.random() * options.length)];
        const nc = choice[0];
        const nr = choice[1];
        const dc = choice[2];
        const dr = choice[3];
        if (dc === 1) vWall[r][c + 1] = false;
        else if (dc === -1) vWall[r][c] = false;
        else if (dr === 1) hWall[r + 1][c] = false;
        else if (dr === -1) hWall[r][c] = false;
        visited[nr][nc] = true;
        stack.push([nc, nr]);
    }
}

function playerCell() {
    return {
        c: clamp(Math.floor(player.x), 0, COLS - 1),
        r: clamp(Math.floor(player.y), 0, ROWS - 1)
    };
}

function obstacleById(id) {
    return obstacles.find((obstacle) => obstacle.id === id) || null;
}

function obstacleAt(c, r) {
    return obstacles.some((obstacle) => obstacle.c === c && obstacle.r === r);
}

/**
 * Open 4-neighbors of a cell. A missing wall is a passage.
 * @param {number} c
 * @param {number} r
 * @returns {number[][]}
 */
function openNeighbors(c, r) {
    const neighbors = [];
    if (c + 1 < COLS && !vWall[r][c + 1]) neighbors.push([c + 1, r]);
    if (c - 1 >= 0 && !vWall[r][c]) neighbors.push([c - 1, r]);
    if (r + 1 < ROWS && !hWall[r + 1][c]) neighbors.push([c, r + 1]);
    if (r - 1 >= 0 && !hWall[r][c]) neighbors.push([c, r - 1]);
    return neighbors;
}

/**
 * True when the shared edge between two orthogonal cells is still a wall.
 * Any other pair is treated as blocked.
 * @param {number} c
 * @param {number} r
 * @param {number} nc
 * @param {number} nr
 * @returns {boolean}
 */
function wallBetween(c, r, nc, nr) {
    if (nc === c + 1 && nr === r) return !!vWall[r][c + 1];
    if (nc === c - 1 && nr === r) return !!vWall[r][c];
    if (nr === r + 1 && nc === c) return !!hWall[r + 1][c];
    if (nr === r - 1 && nc === c) return !!hWall[r][c];
    return true;
}

/**
 * Breadth-first search on open cells. Obstacle cells are closed, so a layout
 * is kept only when the player can still reach the end.
 * @returns {boolean}
 */
function pathExists() {
    const start = playerCell();
    const blocked = new Set(obstacles.map((obstacle) => obstacle.c + "," + obstacle.r));
    if (blocked.has(start.c + "," + start.r) || blocked.has(end.c + "," + end.r)) return false;
    const queue = [[start.c, start.r]];
    const seen = new Set([start.c + "," + start.r]);
    while (queue.length) {
        const current = queue.shift();
        const c = current[0];
        const r = current[1];
        if (c === end.c && r === end.r) return true;
        for (const next of openNeighbors(c, r)) {
            const key = next[0] + "," + next[1];
            if (seen.has(key) || blocked.has(key)) continue;
            seen.add(key);
            queue.push(next);
        }
    }
    return false;
}

function addObstacle(mesh, c, r, live) {
    const obstacle = {
        id: nextId++,
        mesh,
        c,
        r,
        offsetX: 0,
        offsetY: 0,
        quat: quat.create(),
        scale: 0.62,
        live: live || null,
        phase: Math.random() * Math.PI * 2
    };
    obstacles.push(obstacle);
    return obstacle;
}

function tryPlace(mesh, c, r, live) {
    const start = playerCell();
    if (c === start.c && r === start.r) return false;
    if (c === end.c && r === end.r) return false;
    if (obstacleAt(c, r)) return false;
    addObstacle(mesh, c, r, live);
    if (!pathExists()) {
        obstacles.pop();
        return false;
    }
    return true;
}

function placeInitialObstacles() {
    const candidates = [];
    const start = playerCell();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if ((c === start.c && r === start.r) || (c === end.c && r === end.r)) continue;
            if (openNeighbors(c, r).length === 1) candidates.push([c, r]);
        }
    }
    shuffle(candidates);
    const plan = [
        ["airplane", null],
        ["teapot", "spin"],
        ["shark", "patrol"],
        ["ant", null]
    ];
    for (const [mesh, live] of plan) {
        let placed = false;
        while (candidates.length && !placed) {
            const cell = candidates.pop();
            placed = tryPlace(mesh, cell[0], cell[1], live);
        }
        if (!placed) {
            for (let r = 0; r < ROWS && !placed; r++) {
                for (let c = 0; c < COLS && !placed; c++) placed = tryPlace(mesh, c, r, live);
            }
        }
    }
}

function facePlayerIntoMaze() {
    const start = playerCell();
    const next = openNeighbors(start.c, start.r)[0];
    player.pitch = 0;
    player.roll = 0;
    player.yaw = 0;
    if (next) {
        const dc = next[0] - start.c;
        const dr = next[1] - start.r;
        if (dc === -1) player.yaw = Math.PI;
        else if (dr === 1) player.yaw = Math.PI / 2;
        else if (dr === -1) player.yaw = -Math.PI / 2;
    }
    rebuildPlayerQuat();
}

/**
 * Rebuilds the view quaternion from yaw, pitch, and roll.
 * Yaw is around world Z, then pitch around Y, then roll around X.
 * Rebuilding each time keeps the horizon level. Multiplying onto the old
 * quaternion made later turns inherit the tilt, which rolled the view.
 */
function rebuildPlayerQuat() {
    const yaw = quat.create();
    const pitch = quat.create();
    const roll = quat.create();
    quat.setAxisAngle(yaw, [0, 0, 1], player.yaw);
    quat.setAxisAngle(pitch, [0, 1, 0], player.pitch);
    quat.setAxisAngle(roll, [1, 0, 0], player.roll);
    quat.multiply(player.quat, yaw, pitch);
    quat.multiply(player.quat, player.quat, roll);
    quat.normalize(player.quat, player.quat);
}

function lookBy(dyaw, dpitch) {
    player.yaw += dyaw;
    player.pitch = clamp(player.pitch + dpitch, -1.15, 1.15);
    rebuildPlayerQuat();
}

function layoutMaze() {
    generateMaze();
    player.x = 0.5;
    player.y = 0.5;
    player.yaw = 0;
    player.pitch = 0;
    player.roll = 0;
    player.quat = quat.create();
    end = { c: COLS - 1, r: ROWS - 1 };
    obstacles = [];
    selection = null;
    placeInitialObstacles();
    facePlayerIntoMaze();
    setStatus("New maze. The green start still has a path to the gold end.");
}

function circleHitsAabb(x, y, radius, minX, minY, maxX, maxY) {
    const nearestX = clamp(x, minX, maxX);
    const nearestY = clamp(y, minY, maxY);
    const dx = x - nearestX;
    const dy = y - nearestY;
    return dx * dx + dy * dy < radius * radius;
}

/**
 * The player is a circle on the floor. Walls are thin rectangles and each
 * obstacle occupies its whole cell, so the player has to turn back.
 * @param {number} x
 * @param {number} y
 * @returns {boolean}
 */
function canOccupy(x, y) {
    if (x < PLAYER_RADIUS || y < PLAYER_RADIUS || x > COLS - PLAYER_RADIUS || y > ROWS - PLAYER_RADIUS) {
        return false;
    }
    for (let r = 0; r <= ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (!hWall[r][c]) continue;
            if (circleHitsAabb(x, y, PLAYER_RADIUS, c, r - 0.05, c + 1, r + 0.05)) return false;
        }
    }
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c <= COLS; c++) {
            if (!vWall[r][c]) continue;
            if (circleHitsAabb(x, y, PLAYER_RADIUS, c - 0.05, r, c + 0.05, r + 1)) return false;
        }
    }
    for (const obstacle of obstacles) {
        if (circleHitsAabb(x, y, PLAYER_RADIUS, obstacle.c, obstacle.r, obstacle.c + 1, obstacle.r + 1)) {
            return false;
        }
    }
    return true;
}

function horizontalForward() {
    return [Math.cos(player.yaw), Math.sin(player.yaw)];
}

function horizontalRight() {
    return [Math.sin(player.yaw), -Math.cos(player.yaw)];
}

function rotateQuaternion(target, axis, angle) {
    const delta = quat.create();
    quat.setAxisAngle(delta, axis, angle);
    quat.multiply(target, target, delta);
    quat.normalize(target, target);
}

function rotationTarget() {
    if (mode === "play" || (selection && selection.kind === "player")) return player.quat;
    if (selection && selection.kind === "obstacle") {
        const obstacle = obstacleById(selection.id);
        return obstacle ? obstacle.quat : null;
    }
    return null;
}

function rotateSelection(axis, angle) {
    const target = rotationTarget();
    if (!target) {
        setStatus("Select the player or an obstacle before rotating.", "warn");
        return;
    }
    rotateQuaternion(target, axis, angle);
}

function updatePlayer(dt) {
    let turn = 0;
    if (held.has("KeyA")) turn += 1;
    if (held.has("KeyD")) turn -= 1;
    if (turn) player.yaw += turn * 2.1 * dt;

    let pitch = 0;
    if (held.has("KeyQ")) pitch -= 1;
    if (held.has("KeyE")) pitch += 1;
    if (pitch) player.pitch = clamp(player.pitch + pitch * 1.15 * dt, -1.15, 1.15);

    let roll = 0;
    if (held.has("KeyC")) roll += 1;
    if (held.has("KeyV")) roll -= 1;
    if (roll) player.roll = clamp(player.roll + roll * 1.15 * dt, -0.55, 0.55);
    rebuildPlayerQuat();

    let move = 0;
    if (held.has("KeyW")) move += 1;
    if (held.has("KeyS")) move -= 1;
    let strafe = 0;
    if (held.has("KeyF")) strafe += 1;
    if (held.has("KeyG")) strafe -= 1;
    if (!move && !strafe) return;

    const forward = horizontalForward();
    const right = horizontalRight();
    const distance = player.speed * dt;
    const nextX = player.x + (forward[0] * move + right[0] * strafe) * distance;
    const nextY = player.y + (forward[1] * move + right[1] * strafe) * distance;
    if (canOccupy(nextX, nextY)) {
        player.x = nextX;
        player.y = nextY;
    } else if (canOccupy(nextX, player.y)) {
        player.x = nextX;
    } else if (canOccupy(player.x, nextY)) {
        player.y = nextY;
    } else {
        setStatus("Wall or obstacle ahead. Turn back and retrace.", "warn");
        return;
    }
    const cell = playerCell();
    if (cell.c === end.c && cell.r === end.r) setStatus("You reached the end.");
    else if (statusMessage.indexOf("ahead") !== -1) setStatus("Path is open.");
}

function updateScaleAndSpeed(dt) {
    if (held.has("ArrowUp")) player.speed = clamp(player.speed + dt * 0.8, 0.3, 4.5);
    if (held.has("ArrowDown")) player.speed = clamp(player.speed - dt * 0.8, 0.3, 4.5);
    if (mode === "play") return;
    if (!selection || selection.kind !== "obstacle") return;
    const obstacle = obstacleById(selection.id);
    if (!obstacle) return;
    if (held.has("ArrowRight")) obstacle.scale = clamp(obstacle.scale + dt * 0.35, 0.2, 0.9);
    if (held.has("ArrowLeft")) obstacle.scale = clamp(obstacle.scale - dt * 0.35, 0.2, 0.9);
}

function updateLive(dt) {
    for (const obstacle of obstacles) {
        if (obstacle.live === "spin") {
            rotateQuaternion(obstacle.quat, [0, 0, 1], dt * 0.7);
        } else if (obstacle.live === "patrol") {
            obstacle.offsetX = Math.sin(time * 0.9 + obstacle.phase) * 0.16;
        }
    }
}

function aheadCell() {
    const forward = horizontalForward();
    const start = playerCell();
    if (!forward) return { c: start.c, r: start.r, kind: "here" };
    for (const distance of [0.5, 0.95, 1.25]) {
        const x = player.x + forward[0] * distance;
        const y = player.y + forward[1] * distance;
        const c = Math.floor(x);
        const r = Math.floor(y);
        if (c === start.c && r === start.r) continue;
        if (c < 0 || r < 0 || c >= COLS || r >= ROWS) return { c: start.c, r: start.r, kind: "blocked" };
        if (wallBetween(start.c, start.r, c, r) || obstacleAt(c, r)) return { c, r, kind: "blocked" };
        if (c === end.c && r === end.r) return { c, r, kind: "end" };
        return { c, r, kind: "open" };
    }
    return { c: start.c, r: start.r, kind: "here" };
}

function floorColor(c, r) {
    if (mode === "play") {
        const ahead = aheadCell();
        if (ahead.kind !== "here" && ahead.c === c && ahead.r === r) {
            if (ahead.kind === "open") return [0.35, 0.82, 0.9, 1];
            if (ahead.kind === "end") return [1, 0.86, 0.28, 1];
            return [0.9, 0.24, 0.18, 1];
        }
    }
    if (c === end.c && r === end.r) {
        const pulse = 0.82 + 0.18 * Math.sin(time * 3);
        return [0.93 * pulse, 0.68 * pulse, 0.22, 1];
    }
    const start = playerCell();
    if (c === start.c && r === start.r) return [0.45, 0.66, 0.42, 1];
    if ((c + r) % 2 === 0) return [0.8, 0.73, 0.61, 1];
    return [0.74, 0.67, 0.55, 1];
}

function resizeCanvas() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
}

/**
 * Builds the projection and view for the current mode.
 * Top view is orthographic and looks down +Z. World view orbits the maze.
 * Player view is the inverse of the player's camera matrix, so the quaternion
 * is the view orientation.
 * @returns {{projection: mat4, view: mat4}}
 */
function cameraMatrices() {
    const projection = mat4.create();
    const view = mat4.create();
    const aspect = canvas.width / Math.max(1, canvas.height);
    if (mode === "top") {
        const cx = COLS / 2;
        const cy = ROWS / 2;
        let halfW = (COLS + 0.8) / 2;
        let halfH = (ROWS + 0.8) / 2;
        if (aspect > 1) halfW = halfH * aspect;
        else halfH = halfW / aspect;
        floorBounds = {
            left: cx - halfW,
            right: cx + halfW,
            bottom: cy - halfH,
            top: cy + halfH
        };
        // Ortho is in eye space. lookAt already centers the maze, so the box is symmetric.
        mat4.ortho(projection, -halfW, halfW, -halfH, halfH, 0.1, 40);
        mat4.lookAt(view, [cx, cy, 20], [cx, cy, 0], [0, 1, 0]);
        return { projection, view };
    }
    mat4.perspective(projection, (mode === "play" ? 58 : 48) * Math.PI / 180, aspect, 0.08, 80);
    if (mode === "world") {
        const cx = COLS / 2;
        const cy = ROWS / 2;
        const planar = Math.cos(orbitPitch);
        const eye = [
            cx + orbitDist * planar * Math.cos(orbitYaw),
            cy + orbitDist * planar * Math.sin(orbitYaw),
            1.1 + orbitDist * Math.sin(orbitPitch)
        ];
        mat4.lookAt(view, eye, [cx, cy, 0.3], [0, 0, 1]);
        return { projection, view };
    }
    const rotation = mat4.create();
    mat4.fromQuat(rotation, player.quat);
    const basis = mat4.fromValues(
        0, -1, 0, 0,
        0, 0, 1, 0,
        -1, 0, 0, 0,
        0, 0, 0, 1
    );
    const rotated = mat4.create();
    mat4.multiply(rotated, rotation, basis);
    const cameraWorld = mat4.create();
    mat4.fromTranslation(cameraWorld, [player.x, player.y, 0.56]);
    mat4.multiply(cameraWorld, cameraWorld, rotated);
    if (!mat4.invert(view, cameraWorld)) mat4.identity(view);
    return { projection, view };
}

/**
 * Converts a mouse event into floor coordinates for the top view.
 * The division uses the displayed rectangle, not the drawing-buffer size.
 * @param {MouseEvent} event
 * @returns {{x: number, y: number}}
 */
function mouseToFloor(event) {
    const rect = canvas.getBoundingClientRect();
    const u = (event.clientX - rect.left) / rect.width;
    const v = (event.clientY - rect.top) / rect.height;
    return {
        x: floorBounds.left + u * (floorBounds.right - floorBounds.left),
        y: floorBounds.top - v * (floorBounds.top - floorBounds.bottom)
    };
}

function mouseNdc(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
        y: 1 - ((event.clientY - rect.top) / rect.height) * 2
    };
}

/**
 * Projects the mouse onto a unit sphere around the screen center.
 * The z component bulges toward the viewer. Two of these points define the
 * trackball rotation.
 * @param {MouseEvent} event
 * @returns {vec3}
 */
function mouseSphere(event) {
    const ndc = mouseNdc(event);
    const radius2 = ndc.x * ndc.x + ndc.y * ndc.y;
    let z;
    if (radius2 <= 0.5) z = Math.sqrt(Math.max(0, 1 - radius2));
    else z = 0.5 / Math.sqrt(radius2);
    const point = vec3.fromValues(ndc.x, ndc.y, z);
    vec3.normalize(point, point);
    return point;
}

function rayFromMouse(event, projection, view) {
    const ndc = mouseNdc(event);
    const inverse = mat4.create();
    mat4.multiply(inverse, projection, view);
    mat4.invert(inverse, inverse);
    function unproject(depth) {
        const point = vec4.fromValues(ndc.x, ndc.y, depth, 1);
        vec4.transformMat4(point, point, inverse);
        const w = point[3] || 1;
        return [point[0] / w, point[1] / w, point[2] / w];
    }
    const origin = unproject(-1);
    const far = unproject(1);
    const direction = vec3.subtract(vec3.create(), far, origin);
    vec3.normalize(direction, direction);
    return { origin, direction };
}

function rayFloorPoint(event, projection, view) {
    const ray = rayFromMouse(event, projection, view);
    if (Math.abs(ray.direction[2]) < 1e-5) return null;
    const t = -ray.origin[2] / ray.direction[2];
    if (t < 0) return null;
    return [
        ray.origin[0] + ray.direction[0] * t,
        ray.origin[1] + ray.direction[1] * t
    ];
}

function raySphere(origin, direction, center, radius) {
    const offset = vec3.subtract(vec3.create(), origin, center);
    const b = vec3.dot(offset, direction);
    const c = vec3.dot(offset, offset) - radius * radius;
    const discriminant = b * b - c;
    if (discriminant < 0) return null;
    const t = -b - Math.sqrt(discriminant);
    return t > 0 ? t : null;
}

function distToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const length2 = dx * dx + dy * dy;
    let t = length2 ? ((px - ax) * dx + (py - ay) * dy) / length2 : 0;
    t = clamp(t, 0, 1);
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function pickMarker(x, y) {
    let best = null;
    let bestDistance = 0.28;
    function consider(px, py, chosen) {
        const distance = Math.hypot(x - px, y - py);
        if (distance < bestDistance) {
            bestDistance = distance;
            best = chosen;
        }
    }
    consider(player.x, player.y, { kind: "player" });
    consider(end.c + 0.5, end.r + 0.5, { kind: "end" });
    for (const obstacle of obstacles) {
        consider(obstacle.c + 0.5 + obstacle.offsetX, obstacle.r + 0.5 + obstacle.offsetY, {
            kind: "obstacle",
            id: obstacle.id
        });
    }
    return best;
}

function nearestWall(x, y) {
    let best = null;
    let bestDistance = 0.16;
    for (let r = 1; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const distance = distToSegment(x, y, c, r, c + 1, r);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = { kind: "h", r, c };
            }
        }
    }
    for (let r = 0; r < ROWS; r++) {
        for (let c = 1; c < COLS; c++) {
            const distance = distToSegment(x, y, c, r, c, r + 1);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = { kind: "v", r, c };
            }
        }
    }
    return best;
}

function pickWorld(event) {
    const camera = cameraMatrices();
    const ray = rayFromMouse(event, camera.projection, camera.view);
    let best = null;
    let bestT = Infinity;
    function consider(center, radius, chosen) {
        const t = raySphere(ray.origin, ray.direction, center, radius);
        if (t !== null && t < bestT) {
            bestT = t;
            best = chosen;
        }
    }
    consider([player.x, player.y, 0.35], 0.34, { kind: "player" });
    consider([end.c + 0.5, end.r + 0.5, 0.45], 0.36, { kind: "end" });
    for (const obstacle of obstacles) {
        consider(
            [obstacle.c + 0.5 + obstacle.offsetX, obstacle.r + 0.5 + obstacle.offsetY, obstacle.scale * 0.45],
            Math.max(0.3, obstacle.scale * 0.72),
            { kind: "obstacle", id: obstacle.id }
        );
    }
    return best;
}

function toggleWall(wall) {
    const grid = wall.kind === "h" ? hWall : vWall;
    grid[wall.r][wall.c] = !grid[wall.r][wall.c];
    if (!pathExists()) {
        grid[wall.r][wall.c] = !grid[wall.r][wall.c];
        setStatus("That wall stays. Changing it would remove every path to the end.", "warn");
        return;
    }
    const playerPos = { x: player.x, y: player.y };
    if (!canOccupy(player.x, player.y)) {
        grid[wall.r][wall.c] = !grid[wall.r][wall.c];
        player.x = playerPos.x;
        player.y = playerPos.y;
        setStatus("That wall would land on the player.", "warn");
        return;
    }
    setStatus("Wall updated. A path to the end still exists.");
}

function moveSelectionToCell(c, r) {
    if (c < 0 || r < 0 || c >= COLS || r >= ROWS || !selection) return;
    if (selection.kind === "obstacle") {
        const obstacle = obstacleById(selection.id);
        if (!obstacle || (obstacle.c === c && obstacle.r === r)) return;
        const previous = { c: obstacle.c, r: obstacle.r };
        obstacle.c = c;
        obstacle.r = r;
        obstacle.offsetX = 0;
        obstacle.offsetY = 0;
        const start = playerCell();
        const illegal = (c === start.c && r === start.r) || (c === end.c && r === end.r) ||
            obstacles.some((other) => other.id !== obstacle.id && other.c === c && other.r === r) ||
            !pathExists();
        if (illegal) {
            obstacle.c = previous.c;
            obstacle.r = previous.r;
            setStatus("That cell is taken, or it would close every path to the end.", "warn");
        }
        return;
    }
    if (selection.kind === "player") {
        if (obstacleAt(c, r)) {
            setStatus("The player cannot stand on an obstacle.", "warn");
            return;
        }
        const previous = { x: player.x, y: player.y };
        player.x = c + 0.5;
        player.y = r + 0.5;
        if (!pathExists() || !canOccupy(player.x, player.y)) {
            player.x = previous.x;
            player.y = previous.y;
            setStatus("The player has to keep a route to the end.", "warn");
        }
        return;
    }
    if (selection.kind === "end") {
        if (obstacleAt(c, r)) {
            setStatus("The end cannot sit on an obstacle.", "warn");
            return;
        }
        const previous = { c: end.c, r: end.r };
        end.c = c;
        end.r = r;
        if (!pathExists()) {
            end.c = previous.c;
            end.r = previous.r;
            setStatus("The end has to stay reachable.", "warn");
        }
    }
}

/**
 * Multiplies a trackball quaternion into the selected orientation.
 * The sphere axis is in screen space, so it is rewritten with the camera's
 * right, up, and back axes before it touches the object quaternion.
 * @param {vec3} previous
 * @param {vec3} current
 * @param {mat4} view
 */
function applyTrackball(previous, current, view) {
    const axisCamera = vec3.cross(vec3.create(), previous, current);
    const sine = vec3.length(axisCamera);
    const cosine = clamp(vec3.dot(previous, current), -1, 1);
    if (sine < 1e-5) return;
    vec3.scale(axisCamera, axisCamera, 1 / sine);
    const cameraWorld = mat4.create();
    if (!mat4.invert(cameraWorld, view)) return;
    const right = [cameraWorld[0], cameraWorld[1], cameraWorld[2]];
    const up = [cameraWorld[4], cameraWorld[5], cameraWorld[6]];
    const back = [cameraWorld[8], cameraWorld[9], cameraWorld[10]];
    const axis = vec3.fromValues(
        right[0] * axisCamera[0] + up[0] * axisCamera[1] + back[0] * axisCamera[2],
        right[1] * axisCamera[0] + up[1] * axisCamera[1] + back[1] * axisCamera[2],
        right[2] * axisCamera[0] + up[2] * axisCamera[1] + back[2] * axisCamera[2]
    );
    if (vec3.length(axis) < 1e-5) return;
    vec3.normalize(axis, axis);
    const target = rotationTarget();
    if (!target) {
        setStatus("Select the player or an obstacle to use the trackball.", "warn");
        return;
    }
    const delta = quat.create();
    quat.setAxisAngle(delta, axis, Math.atan2(sine, cosine));
    quat.multiply(target, delta, target);
    quat.normalize(target, target);
}

function referencePoint() {
    if (selection && selection.kind === "player") return [player.x, player.y, 0.02];
    if (selection && selection.kind === "end") return [end.c + 0.5, end.r + 0.5, 0.02];
    if (selection && selection.kind === "obstacle") {
        const obstacle = obstacleById(selection.id);
        if (obstacle) return [obstacle.c + 0.5 + obstacle.offsetX, obstacle.r + 0.5 + obstacle.offsetY, 0.02];
    }
    return [0, 0, 0];
}

function referenceQuaternion() {
    if (selection && selection.kind === "player") return player.quat;
    if (selection && selection.kind === "obstacle") {
        const obstacle = obstacleById(selection.id);
        if (obstacle) return obstacle.quat;
    }
    return quat.create();
}

function drawScene(camera) {
    gl.uniformMatrix4fv(projectionLocation, false, camera.projection);
    gl.uniformMatrix4fv(viewLocation, false, camera.view);

    const grout = [0.56, 0.48, 0.39, 1];
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            drawMesh(boxMesh, trs([c + 0.5, r + 0.5, 0.008], null, [1, 1, 0.016]), grout, 1);
            drawMesh(boxMesh, trs([c + 0.5, r + 0.5, 0.024], null, [0.96, 0.96, 0.02]), floorColor(c, r), 1);
        }
    }
    const wallColor = [0.33, 0.24, 0.18, 1];
    const capColor = mode === "top" ? [0.28, 0.2, 0.15, 1] : [0.64, 0.54, 0.42, 1];
    for (let r = 0; r <= ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (!hWall[r][c]) continue;
            drawMesh(boxMesh, trs([c + 0.5, r, WALL_H / 2], null, [1, WALL_T, WALL_H]), wallColor, 1);
            drawMesh(boxMesh, trs([c + 0.5, r, WALL_H + 0.02], null, [1.04, WALL_T + 0.04, 0.05]), capColor, 1);
        }
    }
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c <= COLS; c++) {
            if (!vWall[r][c]) continue;
            drawMesh(boxMesh, trs([c, r + 0.5, WALL_H / 2], null, [WALL_T, 1, WALL_H]), wallColor, 1);
            drawMesh(boxMesh, trs([c, r + 0.5, WALL_H + 0.02], null, [WALL_T + 0.04, 1.04, 0.05]), capColor, 1);
        }
    }

    if (mode === "top") {
        drawMesh(circleMesh, trs([player.x, player.y, 1.12], null, [0.2, 0.2, 1]), [0.16, 0.62, 0.28, 1], 0);
        const forward = horizontalForward() || [1, 0];
        drawMesh(
            boxMesh,
            trs([player.x + forward[0] * 0.28, player.y + forward[1] * 0.28, 1.14], null, [0.14, 0.05, 0.04]),
            [0.08, 0.28, 0.1, 1],
            0
        );
        drawMesh(circleMesh, trs([end.c + 0.5, end.r + 0.5, 1.12], null, [0.2, 0.2, 1]), [0.9, 0.68, 0.12, 1], 0);
        for (const obstacle of obstacles) {
            drawMesh(
                circleMesh,
                trs([obstacle.c + 0.5 + obstacle.offsetX, obstacle.r + 0.5 + obstacle.offsetY, 1.12], null, [0.2, 0.2, 1]),
                MESH_COLOR[obstacle.mesh],
                0
            );
        }
    } else {
        for (const obstacle of obstacles) {
            const mesh = library[obstacle.mesh];
            if (!mesh) continue;
            drawMesh(
                mesh,
                trs(
                    [obstacle.c + 0.5 + obstacle.offsetX, obstacle.r + 0.5 + obstacle.offsetY, 0],
                    obstacle.quat,
                    [obstacle.scale, obstacle.scale, obstacle.scale]
                ),
                MESH_COLOR[obstacle.mesh],
                1
            );
        }
        drawMesh(
            pyramidMesh,
            trs([end.c + 0.5, end.r + 0.5, 0], null, [0.28, 0.28, 0.9]),
            [0.92, 0.7, 0.14, 1],
            1
        );
        if (mode === "world") {
            const upright = quat.create();
            quat.setAxisAngle(upright, [0, 0, 1], player.yaw);
            const body = trs([player.x, player.y, 0], upright, null);
            mat4.translate(body, body, [0.02, 0, 0.24]);
            mat4.scale(body, body, [0.28, 0.16, 0.36]);
            drawMesh(boxMesh, body, [0.2, 0.36, 0.78, 1], 1);
            const head = trs([player.x, player.y, 0], upright, null);
            mat4.translate(head, head, [0.16, 0, 0.5]);
            mat4.scale(head, head, [0.14, 0.14, 0.14]);
            drawMesh(boxMesh, head, [0.35, 0.55, 0.9, 1], 1);
        }
    }

    if (selection && mode !== "play") {
        let x = player.x;
        let y = player.y;
        if (selection.kind === "end") {
            x = end.c + 0.5;
            y = end.r + 0.5;
        } else if (selection.kind === "obstacle") {
            const obstacle = obstacleById(selection.id);
            if (obstacle) {
                x = obstacle.c + 0.5 + obstacle.offsetX;
                y = obstacle.r + 0.5 + obstacle.offsetY;
            }
        }
        drawMesh(
            circleMesh,
            trs([x, y, mode === "top" ? 1.16 : 0.05], null, [0.32, 0.32, 1]),
            [0.95, 0.82, 0.12, 1],
            0
        );
    }

    if (!axesVisible) return;
    const origin = referencePoint();
    const orientation = referenceQuaternion();
    const rods = [
        { dir: [1, 0, 0], color: [0.9, 0.15, 0.12, 1], scale: [0.75, 0.035, 0.035] },
        { dir: [0, 1, 0], color: [0.15, 0.65, 0.2, 1], scale: [0.035, 0.75, 0.035] },
        { dir: [0, 0, 1], color: [0.18, 0.35, 0.95, 1], scale: [0.035, 0.035, 0.75] }
    ];
    const baseZ = mode === "top" ? 1.2 : origin[2];
    for (const rod of rods) {
        const matrix = mat4.create();
        mat4.translate(matrix, matrix, [origin[0], origin[1], baseZ]);
        const rotation = mat4.create();
        mat4.fromQuat(rotation, orientation);
        mat4.multiply(matrix, matrix, rotation);
        mat4.translate(matrix, matrix, [rod.dir[0] * 0.38, rod.dir[1] * 0.38, rod.dir[2] * 0.38]);
        mat4.scale(matrix, matrix, rod.scale);
        drawMesh(boxMesh, matrix, rod.color, 0);
    }
}

function updateHud() {
    let label = "Nothing selected";
    if (mode === "play" || (selection && selection.kind === "player")) label = "Player";
    else if (selection && selection.kind === "end") label = "End marker";
    else if (selection && selection.kind === "obstacle") {
        const obstacle = obstacleById(selection.id);
        label = obstacle ? obstacle.mesh + " obstacle" : "Obstacle";
    }
    const axisNote = axesVisible ? "axes on" : "axes off";
    selectionText.textContent = label + " · speed " + player.speed.toFixed(2) + " · " + obstacles.length + " obstacles · " + axisNote;
}

function draw() {
    resizeCanvas();
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (mode === "top") gl.clearColor(0.86, 0.82, 0.74, 1);
    else gl.clearColor(0.55, 0.66, 0.76, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.uniform1f(fogLocation, mode === "top" ? 0 : mode === "play" ? 0.055 : 0.022);
    gl.uniform3fv(fogColorLocation, mode === "play" ? [0.55, 0.66, 0.76] : [0.62, 0.72, 0.8]);
    drawScene(cameraMatrices());
    updateHud();
}

const MODE_HELP = {
    top: "Top view: click a wall to open or close it. Drag the green player, the gold end, and the colored obstacles.",
    world: "World view: left-drag moves the picked object. Right-drag rotates it with the trackball, or orbits the camera when nothing is selected.",
    play: "Play view: click the maze to capture the mouse, Esc releases it. W and S walk, A and D turn, Q and E look up and down. The floor tile ahead turns cyan, gold, or red."
};

function setMode(next) {
    mode = next;
    modeSelect.value = next;
    modeHelp.textContent = MODE_HELP[next];
    canvas.style.cursor = next === "play" ? "none" : "default";
    if (crosshair) crosshair.hidden = next !== "play";
    if (next !== "play" && document.pointerLockElement === canvas) document.exitPointerLock();
    if (next === "play") {
        selection = { kind: "player" };
        rebuildPlayerQuat();
        setStatus("Click the maze to look around. Esc releases the mouse.");
    }
}

function toggleAxes() {
    axesVisible = !axesVisible;
    axesToggle.textContent = axesVisible ? "Hide axes" : "Show axes";
}

function deleteSelected() {
    if (!selection || selection.kind !== "obstacle") {
        setStatus("Select an obstacle, then delete it.", "warn");
        return;
    }
    obstacles = obstacles.filter((obstacle) => obstacle.id !== selection.id);
    selection = null;
    setStatus("Obstacle removed. That cell is open again.");
}

function addObstacleFromButton() {
    if (obstacles.length >= MAX_OBSTACLES) {
        setStatus("Five obstacles is the maximum. Delete one before adding another.", "warn");
        return;
    }
    const mesh = MESH_ORDER[meshCursor % MESH_ORDER.length];
    const live = mesh === "teapot" ? "spin" : mesh === "shark" ? "patrol" : null;
    const cells = [];
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) cells.push([c, r]);
    }
    shuffle(cells);
    for (const cell of cells) {
        if (!tryPlace(mesh, cell[0], cell[1], live)) continue;
        meshCursor++;
        selection = { kind: "obstacle", id: obstacles[obstacles.length - 1].id };
        setStatus("Added the " + mesh + ". A path to the end still exists.");
        return;
    }
    setStatus("No open cell keeps a path to the end.", "warn");
}

function onMouseDown(event) {
    if (event.button === 2) event.preventDefault();
    pointer = {
        button: event.button,
        lastX: event.clientX,
        lastY: event.clientY,
        moved: false,
        wall: null,
        dragging: false,
        sphere: mouseSphere(event)
    };
    if (mode === "top") {
        const floor = mouseToFloor(event);
        const marker = pickMarker(floor.x, floor.y);
        if (event.button === 0) {
            if (marker) {
                selection = marker;
                pointer.dragging = true;
            } else {
                selection = null;
                pointer.wall = nearestWall(floor.x, floor.y);
            }
        } else if (marker) {
            selection = marker;
        }
        return;
    }
    if (mode === "world" && event.button === 0) {
        selection = pickWorld(event);
        pointer.dragging = !!selection;
    } else if (mode === "world" && event.button === 2) {
        selection = pickWorld(event);
    } else if (mode === "play") {
        selection = { kind: "player" };
        if (canvas.requestPointerLock) canvas.requestPointerLock();
    }
}

function onMouseMove(event) {
    if (!pointer) return;
    const dx = event.clientX - pointer.lastX;
    const dy = event.clientY - pointer.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 3) pointer.moved = true;

    if (mode === "top" && pointer.button === 0 && pointer.dragging) {
        const floor = mouseToFloor(event);
        moveSelectionToCell(Math.floor(floor.x), Math.floor(floor.y));
    } else if (mode === "world" && pointer.button === 0 && pointer.dragging) {
        const camera = cameraMatrices();
        const hit = rayFloorPoint(event, camera.projection, camera.view);
        if (hit) moveSelectionToCell(Math.floor(hit[0]), Math.floor(hit[1]));
    } else if (mode === "play") {
        if (document.pointerLockElement !== canvas) lookBy(-dx * 0.005, dy * 0.004);
    } else if (pointer.button === 2) {
        const current = mouseSphere(event);
        if (rotationTarget()) applyTrackball(pointer.sphere, current, cameraMatrices().view);
        else if (mode === "world") {
            orbitYaw += dx * 0.01;
            orbitPitch = clamp(orbitPitch + dy * 0.008, 0.28, 1.35);
        }
        pointer.sphere = current;
    }
    pointer.lastX = event.clientX;
    pointer.lastY = event.clientY;
}

function onMouseUp() {
    if (pointer && mode === "top" && pointer.button === 0 && pointer.wall && !pointer.moved) {
        toggleWall(pointer.wall);
    }
    pointer = null;
}

function onWheel(event) {
    event.preventDefault();
    const factor = event.deltaY > 0 ? 0.96 : 1.04;
    if (mode !== "play" && selection && selection.kind === "obstacle") {
        const obstacle = obstacleById(selection.id);
        if (obstacle) obstacle.scale = clamp(obstacle.scale * (event.deltaY > 0 ? 0.96 : 1.04), 0.2, 0.9);
        return;
    }
    if (mode === "play") {
        player.speed = clamp(player.speed * (event.deltaY > 0 ? 0.94 : 1.06), 0.3, 4.5);
        return;
    }
    if (mode === "world") orbitDist = clamp(orbitDist / factor, 6, 28);
}

function onKeyDown(event) {
    if (event.target && (event.target.tagName === "INPUT" || event.target.tagName === "SELECT" || event.target.tagName === "TEXTAREA")) {
        return;
    }
    held.add(event.code);
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Backspace", "Space"].indexOf(event.code) !== -1) {
        event.preventDefault();
    }
    if (event.repeat && ["Digit1", "Digit2", "Digit3", "KeyM", "KeyH", "Delete", "Backspace"].indexOf(event.code) !== -1) {
        return;
    }
    if (event.code === "Digit1") setMode("top");
    else if (event.code === "Digit2") setMode("world");
    else if (event.code === "Digit3") setMode("play");
    else if (event.code === "KeyM") setMode(mode === "top" ? "world" : mode === "world" ? "play" : "top");
    else if (event.code === "KeyH") toggleAxes();
    else if (event.code === "Delete" || event.code === "Backspace") deleteSelected();
    else if (event.code === "KeyR" && mode === "play") {
        player.pitch = 0;
        player.roll = 0;
        rebuildPlayerQuat();
        setStatus("View leveled.");
    } else if (event.code === "KeyX" || event.code === "KeyY" || event.code === "KeyZ") {
        const sign = event.shiftKey ? 1 : -1;
        if (mode === "play") {
            const step = sign * 0.1;
            if (event.code === "KeyZ") player.yaw += step;
            else if (event.code === "KeyY") player.pitch = clamp(player.pitch - step, -1.15, 1.15);
            else player.roll = clamp(player.roll + step, -0.55, 0.55);
            rebuildPlayerQuat();
        } else {
            const axis = event.code === "KeyX" ? [1, 0, 0] : event.code === "KeyY" ? [0, 1, 0] : [0, 0, 1];
            rotateSelection(axis, sign * 0.16);
        }
    }
}

function onKeyUp(event) {
    held.delete(event.code);
}

function init() {
    if (!gl) {
        setStatus("This browser did not provide WebGL.", "warn");
        return;
    }
    const vertexSource = document.getElementById("vertex-shader").textContent;
    const fragmentSource = document.getElementById("fragment-shader").textContent;
    const program = createProgram(
        gl,
        createShader(gl, gl.VERTEX_SHADER, vertexSource),
        createShader(gl, gl.FRAGMENT_SHADER, fragmentSource)
    );
    gl.useProgram(program);
    positionLocation = gl.getAttribLocation(program, "a_position");
    normalLocation = gl.getAttribLocation(program, "a_normal");
    modelLocation = gl.getUniformLocation(program, "u_model");
    viewLocation = gl.getUniformLocation(program, "u_view");
    projectionLocation = gl.getUniformLocation(program, "u_projection");
    normalMatrixLocation = gl.getUniformLocation(program, "u_normalMatrix");
    colorLocation = gl.getUniformLocation(program, "u_color");
    litLocation = gl.getUniformLocation(program, "u_lit");
    fogLocation = gl.getUniformLocation(program, "u_fog");
    fogColorLocation = gl.getUniformLocation(program, "u_fogColor");
    gl.uniform1f(fogLocation, 0);
    gl.uniform3fv(fogColorLocation, [0.55, 0.66, 0.76]);
    gl.enableVertexAttribArray(positionLocation);
    gl.enableVertexAttribArray(normalLocation);
    gl.enable(gl.DEPTH_TEST);

    boxMesh = uploadMesh(createBox());
    circleMesh = uploadMesh(createCircle());
    pyramidMesh = uploadMesh(createPyramid());
    for (const name of MESH_ORDER) {
        library[name] = uploadMesh(parsePLY(PLY_TEXT[name]));
        console.log(name, library[name].count / 3, "triangles");
    }

    layoutMaze();
    modeSelect.addEventListener("change", () => setMode(modeSelect.value));
    document.getElementById("newMaze").addEventListener("click", layoutMaze);
    document.getElementById("addObstacle").addEventListener("click", addObstacleFromButton);
    document.getElementById("deleteObstacle").addEventListener("click", deleteSelected);
    axesToggle.addEventListener("click", toggleAxes);
    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mousemove", (event) => {
        if (mode !== "play" || document.pointerLockElement !== canvas) return;
        if (!event.movementX && !event.movementY) return;
        lookBy(-event.movementX * 0.0022, event.movementY * 0.0018);
    });
    window.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", () => held.clear());

    let last = performance.now();
    function frame(now) {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        time += dt;
        if (mode === "play") updatePlayer(dt);
        updateScaleAndSpeed(dt);
        updateLive(dt);
        draw();
        requestAnimationFrame(frame);
    }
    if (location.hash === "#play") setMode("play");
    else if (location.hash === "#world") setMode("world");
    requestAnimationFrame(frame);
}

try {
    init();
} catch (error) {
    console.error(error);
    setStatus(error && error.message ? error.message : "The scene failed to start.", "warn");
}
