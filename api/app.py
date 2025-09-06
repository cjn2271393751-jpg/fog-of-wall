from flask import Flask, request, jsonify
from mangum import Mangum  # 用于兼容 serverless

app = Flask(__name__)

@app.route("/fog-of-wall", methods=["POST"])
def fog_of_wall():
    data = request.get_json()
    # 这里你可以放你原来的逻辑
    return jsonify({"message": "Hello, fog-of-wall!", "data": data})

# 关键：serverless handler
handler = Mangum(app)

# 全局：保存每个 (challenger_id, game_id) 的会话状态
game_states = {}
state_lock = Lock()  # 简单的线程锁，避免并发请求时状态竞争

# ---- 帮助函数 ----
def apply_direction(pos, direction):
    """从 pos (x,y) 按 direction 得到目标坐标。
       坐标系：x 向右增大，y 向下增大（题目 top-left (0,0)）。"""from flask import Flask, request, jsonify
    x, y = pos
    if direction == "N":
        return (x, y - 1)
    if direction == "S":
        return (x, y + 1)
    if direction == "E":
        return (x + 1, y)
    if direction == "W":
        return (x - 1, y)
    return (x, y)

def get_direction(from_pos, to_pos):
    """相邻格子间方向字符串（from_pos -> to_pos）。"""
    fx, fy = from_pos
    tx, ty = to_pos
    dx = tx - fx
    dy = ty - fy
    if dx == 1 and dy == 0:
        return "E"
    if dx == -1 and dy == 0:
        return "W"
    if dx == 0 and dy == 1:
        return "S"
    if dx == 0 and dy == -1:
        return "N"
    return None

def generate_scan_centers(n):
    """
    生成覆盖全图的 5x5 扫描中心点列表（尽量按间隔 5 布局并覆盖边界）。
    对于较小的 n，会选择所有格作为中心（保证覆盖）。
    返回：升序列表 centers_x 和 centers_y（这里我们对 x, y 取相同集合，最后组合）。
    """
    if n <= 5:
        # 小网格：直接使用所有坐标作为中心（扫描重叠但保证覆盖）
        return list(range(0, n))
    centers = list(range(2, n, 5))
    # 确保右/下边界被覆盖：最后再加上 n-3（使得中心到边界距离 <=2）
    edge_center = n - 3
    if edge_center not in centers:
        centers.append(edge_center)
    centers = sorted(set(centers))
    return centers

def bfs_shortest_path(start, goal, known_walls, n):
    """
    在已知的墙壁集合 known_walls 中计算从 start 到 goal 的最短路径（网格）。
    允许经过未知格子（只避开 known_walls）。
    返回路径列表 [(x0,y0),(x1,y1), ...] 或 None（不可达）。
    允许穿过其他乌鸦占据的格子（题目允许）。
    """
    if start == goal:
        return [start]
    q = deque()
    q.append((start, [start]))
    visited = set([start])
    while q:
        (cx, cy), path = q.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < n and 0 <= ny < n:
                if (nx, ny) in known_walls:
                    continue
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                if (nx, ny) == goal:
                    return new_path
                q.append(((nx, ny), new_path))
    return None

# ---- 状态管理与决策 ----
def init_game_state(test_case):
    """根据 test_case 初始化一个新的 game state。返回 state dict。"""
    length_of_grid = test_case["length_of_grid"]
    num_of_walls = test_case.get("num_of_walls", None)
    crows_list = test_case["crows"]  # 列表，含 id,x,y
    crows = [str(c["id"]) for c in crows_list]  # 统一用字符串 id
    crow_positions = {str(c["id"]):(c["x"], c["y"]) for c in crows_list}
    known_walls = set()
    # 生成扫描中心（覆盖策略）
    centers = generate_scan_centers(length_of_grid)
    scan_centers = [(x, y) for x in centers for y in centers]
    scan_centers.sort(key=lambda p: (p[0], p[1]))
    # 任务分配（平均分配扫描中心给各乌鸦）
    tasks = {cid: [] for cid in crows}
    n = len(crows)
    if n == 0:
        raise ValueError("test_case.crows 不能为空")
    if n == 1:
        tasks[crows[0]] = scan_centers.copy()
    else:
        # 简单的分段分配，保持连续块（也可按 round-robin）
        L = len(scan_centers)
        if n == 2:
            mid = L // 2
            tasks[crows[0]] = scan_centers[:mid]
            tasks[crows[1]] = scan_centers[mid:]
        elif n == 3:
            t = L // 3
            tasks[crows[0]] = scan_centers[:t]
            tasks[crows[1]] = scan_centers[t:2*t]
            tasks[crows[2]] = scan_centers[2*t:]
        else:
            # fallback：round-robin
            for i, cen in enumerate(scan_centers):
                tasks[crows[i % n]].append(cen)
    # current_targets：每只乌鸦的当前目标（tasks[0]）
    current_targets = {cid:(tasks[cid][0] if tasks[cid] else None) for cid in crows}
    state = {
        "length_of_grid": length_of_grid,
        "num_of_walls": num_of_walls,
        "crows": crows,
        "crow_positions": crow_positions,
        "known_walls": known_walls,
        "tasks": tasks,
        "current_targets": current_targets,
        "last_crow_idx": 0,   # 用于轮转选择下一个乌鸦
        "move_count": 0,
        "max_moves": length_of_grid * length_of_grid  # 提示性上限
    }
    return state

def pick_next_crow(state):
    """按轮转选择下一个仍有任务的乌鸦 id；若都无任务返回 None。"""
    crows = state["crows"]
    n = len(crows)
    if n == 0:
        return None
    start = state.get("last_crow_idx", 0) % n
    for i in range(n):
        idx = (start + i) % n
        cid = crows[idx]
        # 有任务或当前目标存在时选择
        if state["current_targets"].get(cid) is not None:
            state["last_crow_idx"] = (idx + 1) % n
            return cid
    return None

def decide_next_action(state, challenger_id, game_id):
    """
    根据当前 state 决策下一步动作。
    返回一个 dict，已经包含 challenger_id 与 game_id（除 submit 情况外也包含 crow_id）。
    """
    # 终止条件：已识别墙数达到题目给定的 num_of_walls（若 num_of_walls 提供）
    if state.get("num_of_walls") is not None:
        if len(state["known_walls"]) >= state["num_of_walls"]:
            # 提交
            submission = [f"{x}-{y}" for x,y in sorted(state["known_walls"])]
            return {"challenger_id": challenger_id, "game_id": game_id,
                    "action_type": "submit", "submission": submission}
    # 终止条件：所有任务都为空
    all_none = all(state["current_targets"].get(cid) is None for cid in state["crows"])
    if all_none:
        submission = [f"{x}-{y}" for x,y in sorted(state["known_walls"])]
        return {"challenger_id": challenger_id, "game_id": game_id,
                "action_type": "submit", "submission": submission}
    # 选择一个乌鸦执行（轮转）
    cid = pick_next_crow(state)
    if cid is None:
        submission = [f"{x}-{y}" for x,y in sorted(state["known_walls"])]
        return {"challenger_id": challenger_id, "game_id": game_id,
                "action_type": "submit", "submission": submission}
    target = state["current_targets"].get(cid)
    if target is None:
        # 没目标（应已在上面被检测），递归决定
        return decide_next_action(state, challenger_id, game_id)
    # 如果已在目标格，发出 scan
    if state["crow_positions"][cid] == target:
        # 计数（每次返回动作都视为一次 move/scan）
        state["move_count"] += 1
        return {"challenger_id": challenger_id, "game_id": game_id,
                "crow_id": cid, "action_type": "scan"}
    # 否则规划路径并移动一步
    path = bfs_shortest_path(state["crow_positions"][cid], target, state["known_walls"], state["length_of_grid"])
    if path is None:
        # 无法到达：放弃该目标（可能被已知墙围住）
        if state["tasks"].get(cid):
            state["tasks"][cid].pop(0)
        state["current_targets"][cid] = state["tasks"][cid][0] if state["tasks"][cid] else None
        return decide_next_action(state, challenger_id, game_id)
    if len(path) <= 1:
        # 理论上已在目标或无法前进，回到上面处理
        state["current_targets"][cid] = state["tasks"][cid][0] if state["tasks"][cid] else None
        return decide_next_action(state, challenger_id, game_id)
    # 下一步移动
    nxt = path[1]
    direction = get_direction(state["crow_positions"][cid], nxt)
    state["move_count"] += 1
    return {"challenger_id": challenger_id, "game_id": game_id,
            "crow_id": cid, "action_type": "move", "direction": direction}

# ---- 输入处理 ----
def process_previous_action(state, prev):
    """
    根据 judge 返回的 previous_action 更新本地 state。
    previous_action 可能包含：
      - your_action: "move" or "scan"
      - crow_id
      - direction (若 move)
      - move_result (若 move) -> [x,y]
      - scan_result (若 scan) -> 5x5 列表
    注意：若 move_result 与先前位置相同，视为撞墙并将目标格标为墙。
    """
    if not prev:
        return
    your_action = prev.get("your_action") or prev.get("action_type")
    cid = str(prev.get("crow_id")) if prev.get("crow_id") is not None else None

    # 若是 move
    if your_action == "move":
        if cid is None:
            return
        prior_pos = state["crow_positions"].get(cid)
        # move_result 可能是列表 [x,y]
        move_res = prev.get("move_result")
        direction = prev.get("direction") or prev.get("dir")  # 容错
        if move_res is not None and prior_pos is not None:
            try:
                rx, ry = int(move_res[0]), int(move_res[1])
            except Exception:
                # 不合法结果，忽略
                rx, ry = prior_pos
            # 如果没有移动（返回位置与 prior 相同），则说明撞到了墙
            if (rx, ry) == prior_pos and direction:
                intended = apply_direction(prior_pos, direction)
                ix, iy = intended
                if 0 <= ix < state["length_of_grid"] and 0 <= iy < state["length_of_grid"]:
                    state["known_walls"].add((ix, iy))
                # 位置保持不变
                state["crow_positions"][cid] = prior_pos
            else:
                # 移动成功，更新位置
                state["crow_positions"][cid] = (rx, ry)
        else:
            # 没有 move_result -> 无更新
            pass

    # 若是 scan
    elif your_action == "scan":
        if cid is None:
            return
        scan_res = prev.get("scan_result")
        if scan_res and isinstance(scan_res, list) and len(scan_res) == 5:
            # 中心在 [2][2]（题目保证）
            cx, cy = state["crow_positions"].get(cid, (None, None))
            if cx is None:
                return
            for dy in range(5):
                for dx in range(5):
                    val = scan_res[dy][dx]
                    gx = cx - 2 + dx
                    gy = cy - 2 + dy
                    if 0 <= gx < state["length_of_grid"] and 0 <= gy < state["length_of_grid"]:
                        if val == "W":
                            state["known_walls"].add((gx, gy))
                        # 我们不强制记录 '_' 为空（冗余），只记录墙可避免误判
            # 如果该扫描是在任务目标上，弹出任务
            cur_target = state["current_targets"].get(cid)
            if cur_target is not None and state["crow_positions"].get(cid) == cur_target:
                if state["tasks"].get(cid):
                    # 弹出已完成的任务（假设它是列表头）
                    try:
                        state["tasks"][cid].pop(0)
                    except Exception:
                        pass
                state["current_targets"][cid] = state["tasks"][cid][0] if state["tasks"][cid] else None
        else:
            # scan_result 缺失或不符合 5x5，忽略
            pass
    else:
        # 其他类型（忽略）
        return

# ---- HTTP 路由 ----
@app.route("/fog-of-wall", methods=["POST"])
def fog_of_wall():
    """
    接收 judge 的请求（初始化或带 previous_action）。
    返回下一步动作。
    """
    data = request.get_json()
    if data is None:
        return jsonify({"error": "invalid json"}), 400

    challenger_id = str(data.get("challenger_id", "unknown"))
    game_id = str(data.get("game_id", "unknown"))
    key = (challenger_id, game_id)

    with state_lock:
        # 如果包含 test_case -> 初始化新的 game state（覆盖同 game 的旧状态）
        if "test_case" in data and data["test_case"]:
            try:
                state = init_game_state(data["test_case"])
            except Exception as e:
                return jsonify({"error": f"invalid test_case: {e}"}), 400
            game_states[key] = state
        else:
            # 非首次请求：必须存在先前 state
            if key not in game_states:
                return jsonify({"error": "unknown game session; send initial test_case first"}), 400
            state = game_states[key]
            prev = data.get("previous_action")
            process_previous_action(state, prev)

        # 决策下一步
        action = decide_next_action(state, challenger_id, game_id)

        # 若返回 submit，我们也保留该 state（或可以选择删除）
        if action.get("action_type") == "submit":
            # 可选：清理 state，释放内存（注释掉，保留以便调试）
            # del game_states[key]
            return jsonify(action)

        # 否则 move 或 scan：确保字段符合题目格式
        # action 已包含 challenger_id 和 game_id
        if action["action_type"] == "move":
            # 必须包含 crow_id, direction
            return jsonify({
                "challenger_id": action["challenger_id"],
                "game_id": action["game_id"],
                "crow_id": action["crow_id"],
                "action_type": "move",
                "direction": action["direction"]
            })
        elif action["action_type"] == "scan":
            return jsonify({
                "challenger_id": action["challenger_id"],
                "game_id": action["game_id"],
                "crow_id": action["crow_id"],
                "action_type": "scan"
            })
        else:
            # 兜底
            return jsonify({"error": "unknown decision"}), 500

# ---- 启动 ----
if __name__ == "__main__":
    # 开发用： debug=True（部署时关闭）

    app.run(host="0.0.0.0", port=8000, debug=True)
