export function buildMatmulShader(tileSize = 16) {
  return `
const TILE: u32 = ${tileSize}u;

struct Params {
  m: u32,
  n: u32,
  k: u32,
}

@group(0) @binding(0) var<storage, read> leftMatrix: array<f32>;
@group(0) @binding(1) var<storage, read> rightMatrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputMatrix: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> leftTile: array<f32, ${tileSize * tileSize}>;
var<workgroup> rightTile: array<f32, ${tileSize * tileSize}>;

@compute @workgroup_size(${tileSize}, ${tileSize}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(global_invocation_id) globalId: vec3<u32>
) {
  let row = globalId.y;
  let col = globalId.x;

  if (row >= params.m || col >= params.n) {
    return;
  }

  let localRow = localId.y;
  let localCol = localId.x;
  var total = 0.0;

  let tileCount = (params.k + TILE - 1u) / TILE;
  for (var tile = 0u; tile < tileCount; tile = tile + 1u) {
    let kLeft = tile * TILE + localCol;
    let kRight = tile * TILE + localRow;

    let leftIndex = row * params.k + kLeft;
    let rightIndex = kRight * params.n + col;
    let sharedIndex = localRow * TILE + localCol;

    leftTile[sharedIndex] = select(0.0, leftMatrix[leftIndex], kLeft < params.k);
    rightTile[sharedIndex] = select(0.0, rightMatrix[rightIndex], kRight < params.k);
    workgroupBarrier();

    for (var inner = 0u; inner < TILE; inner = inner + 1u) {
      total = total + leftTile[localRow * TILE + inner] * rightTile[inner * TILE + localCol];
    }
    workgroupBarrier();
  }

  outputMatrix[row * params.n + col] = total;
}
`;
}

export function buildRmsNormShader() {
  return `
struct Params {
  rows: u32,
  cols: u32,
  epsilon_bits: u32,
}

@group(0) @binding(0) var<storage, read> inputData: array<f32>;
@group(0) @binding(1) var<storage, read> weightData: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputData: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn epsilon() -> f32 {
  return bitcast<f32>(params.epsilon_bits);
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let row = workgroupId.x;
  if (row >= params.rows) {
    return;
  }

  var sumSquares = 0.0;
  for (var col = localId.x; col < params.cols; col = col + 64u) {
    let value = inputData[row * params.cols + col];
    sumSquares = sumSquares + value * value;
  }

  var<workgroup> sharedSums: array<f32, 64>;
  sharedSums[localId.x] = sumSquares;
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (localId.x < stride) {
      sharedSums[localId.x] = sharedSums[localId.x] + sharedSums[localId.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }

  let scale = inverseSqrt(sharedSums[0] / f32(params.cols) + epsilon());
  for (var col = localId.x; col < params.cols; col = col + 64u) {
    let index = row * params.cols + col;
    outputData[index] = inputData[index] * scale * weightData[col];
  }
}
`;
}
