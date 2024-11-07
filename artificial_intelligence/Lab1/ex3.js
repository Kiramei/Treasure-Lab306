class Algorithm {
  static cantor(nums, size) {
    let ans = 0,
      sum = 0;
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) if (nums[j] < nums[i]) sum++;
      ans += sum * factorial(size - i - 2);
      sum = 0;
    }
    return ans;
  }

  static combine(m, n) {
    if (m < n) return 0;
    return factorial(m) / (factorial(n) * factorial(m - n));
  }

  static combineIndex(nums, m, n) {
    let ret = 0;
    for (let i = m - 1; i >= 0; i--)
      if (nums[i]) ret += Algorithm.combine(m - 1 - i, n--);
    return ret;
  }
}

class PrepareSearch {
  constructor() {
    this.init();
  }

  init() {
    this.phase2_fill_buffer(
      cube,
      this.ud_edges_perm,
      UD_EDGES_PERM_SIZE,
      ud_edges_perm_index
    );
    this.phase2_fill_buffer(
      cube,
      this.middle_edges_perm,
      MIDDLE_EDGES_PERM_SIZE,
      middle_edges_perm_index
    );
    this.phase2_fill_buffer(
      cube,
      this.corners_perm,
      CORNORS_PERM_SIZE,
      cornors_perm_index
    );
    this.phase1_fill_buffer(
      cube,
      this.middle_edges_perm_orientation,
      MIDDLE_EDGES_PERM_ORIENTATION_SIZE,
      me_combine_index
    );
    this.phase1_fill_buffer(
      cube,
      this.cornors_orientation,
      CORNORS_ORIENTATION_SIZE,
      co_index
    );
    this.phase1_fill_buffer(
      cube,
      this.edges_orientation,
      EDGES_ORIENTATION_SIZE,
      eo_index
    );
    this.phase2_fill_pre();
    this.phase1_fill_pre();
  }

  calculateIndex(cube, type) {
    switch (type) {
      case co_index:
        return cube.co.reduce((a, b) => a * 3 + b, 0);
      case eo_index:
        return cube.eo.reduce((a, b) => a * 2 + b, 0);
      case me_combine_index: {
        let comArr = cube.ep.map((x) => (x >= 8 ? 1 : 0)),
          permArr = comArr.filter((x) => x).map((x) => cube.ep[x] - 8);
        return (
          Algorithm.combineIndex(comArr, 12, 4) * 24 +
          Algorithm.cantor(permArr, 4)
        );
      }
      case ud_edges_perm_index:
        return Algorithm.cantor(cube.ep.slice(0, 8), 8);
      case middle_edges_perm_index:
        return Algorithm.cantor(
          cube.ep.slice(8, 12).map((x) => x - 8),
          4
        );
      case cornors_perm_index:
        return Algorithm.cantor(cube.cp, 8);
    }
  }

  phase2_fill_buffer(cube, dest, destSize, type) {
    dest.fill(-1);
    let q = [{ cube, step: 0 }];
    dest[0] = 0;
    if (type === middle_edges_perm_index) this.middle_edges_perms.push(cube);
    while (q.length) {
      let { cube, step } = q.shift();
      let fatherIndex = this.calculateIndex(cube, type);
      for (let move = 0; move < 18; move++)
        if (MOVELIMIT & (1 << move)) {
          let current = this.cubeState.moveRotate(move, cube);
          let Index = this.calculateIndex(current, type);
          if (dest[Index] === -1) {
            dest[Index] = step + 1;
            q.push({ cube: current, step: step + 1 });
          }
          if (type === cornors_perm_index)
            this.corners_perm_move[fatherIndex][move] = Index;
          else if (type === ud_edges_perm_index)
            this.ud_edges_perm_move[fatherIndex][move] = Index;
          else if (type === middle_edges_perm_index)
            this.middle_edges_perm_move[fatherIndex][move] = Index;
        }
    }
  }

  phase2FillPre() {
    const MIDDLE_EDGES_PERM_SIZE = 24;
    const CORNORS_PERM_SIZE = 24;
    const MOVELIMIT = 0x3ffff;

    const cp_mep = new Int8Array(
      MIDDLE_EDGES_PERM_SIZE * CORNORS_PERM_SIZE
    ).fill(-1);
    const queue = [];
    let first_t = { corners: 0, edges2: 0 };
    queue.push([first_t, 0]);
    cp_mep[0] = 0;

    while (queue.length > 0) {
      const [front, depth] = queue.shift();
      for (let move = 0; move < 18; move++) {
        if (MOVELIMIT & (1 << move)) {
          const new_t = {
            corners: cornerPermutationMove(front.corners, move),
            edges2: middleEdgesPermutationMove(front.edges2, move),
          };
          const index = new_t.corners * 24 + new_t.edges2;
          if (cp_mep[index] === -1) {
            queue.push([new_t, depth + 1]);
            cp_mep[index] = depth + 1;
          }
        }
      }
    }

    const ep_mep = new Int8Array(MIDDLE_EDGES_PERM_SIZE * 24).fill(-1);
    const queue2 = [];
    let first_t2 = { edges1: 0, edges2: 0 };
    queue2.push([first_t2, 0]);
    ep_mep[0] = 0;
    while (queue2.length > 0) {
      const [front, depth] = queue2.shift();
      for (let move = 0; move < 18; move++) {
        if (MOVELIMIT & (1 << move)) {
          const new_t = {
            edges1: udEdgesPermutationMove(front.edges1, move),
            edges2: middleEdgesPermutationMove(front.edges2, move),
          };
          const index = new_t.edges1 * 24 + new_t.edges2;
          if (ep_mep[index] === -1) {
            queue2.push([new_t, depth + 1]);
            ep_mep[index] = depth + 1;
          }
        }
      }
    }
  }

  phase1FillBuffer(goalstate, dest, destSize, type) {
    const queue = [];
    dest.fill(-1);
    queue.push([goalstate, 0]);
    dest[0] = 0;
    if (type === "me_combine_index") {
      for (let i = 0; i < MIDDLE_EDGES_PERM_SIZE; i++) {
        queue.push([middleEdgesPerms[i], 0]);
        const Index = calculateIndex(middleEdgesPerms[i], type);
        dest[Index] = 0;
      }
    }

    while (queue.length > 0) {
      const [front, fatherIndex] = queue.shift();
      const Index = calculateIndex(front, type);
      for (let move = 0; move < 18; move++) {
        const step = fatherIndex + 1;
        const currstate = moveRotate(move, front);
        const Index = calculateIndex(currstate, type);
        if (type === "co_index") {
          cornorsOrientationMove[fatherIndex][move] = Index;
        } else if (type === "eo_index") {
          edgesOrientationMove[fatherIndex][move] = Index;
        } else {
          middleEdgesPermOrientationMove[fatherIndex][move] = Index;
        }
        if (dest[Index] === -1) {
          dest[Index] = step;
          queue.push([currstate, step]);
        }
      }
    }
  }

  phase1FillPre() {
    const CORNORS_ORIENTATION_SIZE = 2187;
    const MIDDLE_EDGES_COMBINATION = 24;
    const EDGES_ORIENTATION_SIZE = 2048;

    const co_mec = new Int8Array(
      CORNORS_ORIENTATION_SIZE * MIDDLE_EDGES_COMBINATION
    ).fill(-1);
    const queue = [];
    let first_t = { co: 0, middle_edges_combination: 0 };
    queue.push([first_t, 0]);
    co_mec[0] = 0;
    while (queue.length > 0) {
      const [front, depth] = queue.shift();
      for (let move = 0; move < 18; move++) {
        const new_t = {
          co: cornorsOrientationMove[front.co][move],
          middle_edges_combination:
            middleEdgesPermOrientationMove[front.middle_edges_combination][
              move
            ],
        };
        const index =
          new_t.co * MIDDLE_EDGES_COMBINATION +
          Math.floor(new_t.middle_edges_combination / 24);
        if (co_mec[index] === -1) {
          queue.push([new_t, depth + 1]);
          co_mec[index] = depth + 1;
        }
      }
    }

    const eo_mec = new Int8Array(
      EDGES_ORIENTATION_SIZE * MIDDLE_EDGES_COMBINATION
    ).fill(-1);
    const queue2 = [];
    let first_t2 = { eo: 0, middle_edges_combination: 0 };
    queue2.push([first_t2, 0]);
    eo_mec[0] = 0;
    while (queue2.length > 0) {
      const [front, depth] = queue2.shift();
      for (let move = 0; move < 18; move++) {
        const new_t = {
          eo: edgesOrientationMove[front.eo][move],
          middle_edges_combination:
            middleEdgesPermOrientationMove[front.middle_edges_combination][
              move
            ],
        };
        const index =
          new_t.eo * MIDDLE_EDGES_COMBINATION +
          Math.floor(new_t.middle_edges_combination / 24);
        if (eo_mec[index] === -1) {
          queue2.push([new_t, depth + 1]);
          eo_mec[index] = depth + 1;
        }
      }
    }
  }

  static DFSphase2(se_t) {
    for (let move = 0; move < 18; move++) {
      if (prepareSearch.MOVELIMIT & (1 << move)) {
        if (Math.floor(move / 3) === se_t.face) continue;
        const ud_edges_perm_index =
          prepareSearch.ud_edges_perm_move[se_t.ud_edges_perm_index][move];
        const middle_edges_perm_index =
          prepareSearch.middle_edges_perm_move[se_t.middle_edges_perm_index][
            move
          ];
        const cornors_perm_index =
          prepareSearch.corners_perm_move[se_t.cornors_perm_index][move];
        const val = Math.max(
          prepareSearch.ep_mep[
            ud_edges_perm_index * 24 + middle_edges_perm_index
          ],
          prepareSearch.cp_mep[
            cornors_perm_index * 24 + middle_edges_perm_index
          ]
        );
        if (val + se_t.current_depth < se_t.total_depth) {
          se_t.steps.steps[se_t.current_depth] = move;
          if (val === 0) {
            return true;
          }
          const newSe_t = { ...se_t };
          newSe_t.current_depth += 1;
          newSe_t.ud_edges_perm_index = ud_edges_perm_index;
          newSe_t.middle_edges_perm_index = middle_edges_perm_index;
          newSe_t.cornors_perm_index = cornors_perm_index;
          newSe_t.face = Math.floor(move / 3);
          if (prepareSearch.DFSphase2(newSe_t)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  static phase2(cube, moves) {
    for (let i = 0; i < moves.vaildLength; i++) {
      cube = prepareSearch.cubeState.moveRotate(moves.steps[i], cube);
    }
    let phase2_len =
      prepareSearch.MAX_STEP - moves.vaildLength > 10
        ? 10
        : prepareSearch.MAX_STEP - moves.vaildLength;
    for (let depth = 0; depth <= phase2_len; depth++) {
      const moves2 = new prepareSearch.moves_t(depth);
      const search = {
        face: -1,
        ud_edges_perm_index: prepareSearch.calculateIndex(
          cube,
          prepareSearch.ud_edges_perm_index
        ),
        middle_edges_perm_index: prepareSearch.calculateIndex(
          cube,
          prepareSearch.middle_edges_perm_index
        ),
        cornors_perm_index: prepareSearch.calculateIndex(
          cube,
          prepareSearch.cornors_perm_index
        ),
        current_depth: 0,
        total_depth: depth,
        steps: moves2,
      };
      if (prepareSearch.DFSphase2(search)) {
        prepareSearch.printSolution(moves);
        prepareSearch.printSolution(moves2);
        return true;
      }
    }
    return false;
  }

  static DFSphase1(se_t) {
    for (let move = 0; move < 18; move++) {
      if (Math.floor(move / 3) === se_t.face) continue;

      const co_index =
        prepareSearch.cornors_orientation_move[se_t.co_index][move];
      const eo_index =
        prepareSearch.edges_orientation_move[se_t.eo_index][move];
      const me_combine_index =
        prepareSearch.middle_edges_perm_orientation_move[se_t.me_combine_index][
          move
        ];

      const val = Math.max(
        prepareSearch.co_mec[
          prepareSearch.MIDDLE_EDGES_COMBINATION * co_index +
            Math.floor(me_combine_index / 24)
        ],
        prepareSearch.eo_mec[
          prepareSearch.MIDDLE_EDGES_COMBINATION * eo_index +
            Math.floor(me_combine_index / 24)
        ]
      );
      if (val + se_t.current_depth < se_t.total_depth) {
        se_t.steps.steps[se_t.current_depth] = move;
        if (val === 0) {
          if (prepareSearch.phase2(se_t.initstate, se_t.steps)) {
            return true;
          }
        }
        const newSe_t = { ...se_t };
        newSe_t.current_depth += 1;
        newSe_t.face = Math.floor(move / 3);
        newSe_t.co_index = co_index;
        newSe_t.eo_index = eo_index;
        newSe_t.me_combine_index = me_combine_index;
        if (prepareSearch.DFSphase1(newSe_t)) {
          return true;
        }
      }
    }
    return false;
  }

  static solve(cube) {
    let depth = 0;
    while (true) {
      const moves = new prepareSearch.moves_t(depth);
      const search = {
        face: -1,
        initstate: cube,
        current_depth: 0,
        total_depth: depth,
        steps: moves,
        co_index: prepareSearch.calculateIndex(cube, prepareSearch.co_index),
        eo_index: prepareSearch.calculateIndex(cube, prepareSearch.eo_index),
        me_combine_index: prepareSearch.calculateIndex(
          cube,
          prepareSearch.me_combine_index
        ),
      };
      if (prepareSearch.DFSphase1(search)) {
        break;
      }
      depth++;
    }
  }

  static printSolution(s) {
    let output = "";
    for (let i = 0; i < s.vaildLength; i++) {
      output += " ";
      const movesteps = (s.steps[i] % 3) + 1;
      if (movesteps === 3) {
        output += `${prepareSearch.UDFBLR[Math.floor(s.steps[i] / 3)]}'`;
      } else if (movesteps === 1) {
        output += prepareSearch.UDFBLR[Math.floor(s.steps[i] / 3)];
      } else {
        output += `${
          prepareSearch.UDFBLR[Math.floor(s.steps[i] / 3)]
        }${movesteps}`;
      }
    }
    console.log(output);
  }
}
