import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements

def check_loaded_data(plt, data, index=0):
    agents = np.concatenate([data['obj_trajs'][..., :2], data['obj_trajs_future_state'][..., :2]], axis=-2)
    map = data['map_polylines']

    if len(agents.shape) == 4:
        agents = agents[index]
        map = map[index]
        ego_index = data['track_index_to_predict'][index]
        ego_agent = agents[ego_index]
    else:
        ego_index = data['track_index_to_predict']
        ego_agent = agents[ego_index]

    def draw_line_with_mask(point1, point2, color, line_width=4):
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Plot the map with mask check
    for lane in map:
        # map_one_hot = lane[0, -20:]
        # if np.argmax(map_one_hot) in [1, 2, 3]:
        #     continue
        for i in range(len(lane) - 1):
            draw_line_with_mask(lane[i, :2], lane[i, 6:8], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)

    # Set labels, limits, and other properties
    #vis_range = 100
    # plt.xlim(-vis_range + 30, vis_range + 30)
    # plt.ylim(-vis_range, vis_range)
   # plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.axis('equal')
    #plt.tight_layout()

    return plt


def visualize_batch_data(ax, data):
    def decode_obj_trajs(obj_trajs):
        obj_trajs_xy = obj_trajs[..., :2]
        obj_lw = obj_trajs[...,-1, 3:5]
        obj_type_onehot = obj_trajs[...,-1, 6:9]
        obj_type = np.argmax(obj_type_onehot, axis=-1)
        obj_heading_encoding = obj_trajs[...,-1, 33:35]
        return obj_trajs_xy, obj_lw, obj_type, obj_heading_encoding
    def decode_map(map):
        map_xy = map[..., :2]
        map_type = map[...,0, 9:29]
        map_type = np.argmax(map_type, axis=-1)
        return map_xy, map_type

    def plot_objects(obj_xy, obj_lw, obj_heading, obj_mask):
        # 在已有的ax对象上进行绘制
        for i in range(len(obj_lw)):
            if obj_mask[i]:
                # 获取对象的长和宽
                length, width = obj_lw[i]

                # 通过sin和cos计算旋转角度
                sin_angle, cos_angle = obj_heading[i]
                angle = np.arctan2(sin_angle, cos_angle)  # 转换为角度（弧度）

                # 获取对象的中心位置
                x, y = obj_xy[i]

                # 创建旋转矩形对象
                rect = plt.Rectangle((-length / 2, -width / 2), length, width, angle=0,
                                     facecolor='none', edgecolor='grey', linewidth=1)

                # 使用转换矩阵将矩形旋转并平移到对象的中心位置
                t = ax.transData
                # 先将矩形平移到中心位置，然后旋转
                rot = plt.matplotlib.transforms.Affine2D().rotate_around(0, 0, angle).translate(x, y) + t
                rect.set_transform(rot)

                # 添加矩形到现有ax中
                ax.add_patch(rect)
    def draw_trajectory(trajectory, line_width, ego=False):
        def interpolate_color(start_color, end_color, t, total_t):
            """根据 t 和 total_t 插值计算颜色."""
            return [(1 - t / total_t) * start + (t / total_t) * end for start, end in zip(start_color, end_color)]

        def draw_line_with_mask(point1, point2, color, line_width=4):
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color,alpha=0.5)
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                # 天蓝色渐变：从深蓝到浅蓝
                start_color = (0, 0, 0.5)  # 深蓝色
                end_color = (0.53, 0.81, 0.98)  # 浅蓝色
            else:
                # 草绿色渐变：从深绿到浅绿
                start_color = (0, 0.5, 0)  # 深绿色
                end_color = (0.56, 0.93, 0.56)  # 浅绿色

            # 计算当前时间步的颜色
            color = interpolate_color(start_color, end_color, t, total_t)

            if trajectory[t, 0] and trajectory[t + 1, 0]:
                draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    obj_trajs = data['obj_trajs']
    map = data['map_polylines']

    obj_trajs_xy, obj_lw, obj_type, obj_heading = decode_obj_trajs(obj_trajs)
    obj_trajs_future_state = data['obj_trajs_future_state'][...,:2]
    all_traj = np.concatenate([obj_trajs_xy, obj_trajs_future_state], axis=-2)

    for i in range(obj_trajs.shape[0]):
        if i == data['track_index_to_predict']:
            ego = True
        else:
            ego = False
        draw_trajectory(all_traj[i], line_width=3,ego=ego)

    map_xy, map_type = decode_map(map)
    obj_mask = data['obj_trajs_mask']
    plot_objects(obj_trajs_xy[:,-1],obj_lw, obj_heading, obj_mask[:,-1])

    for indx, type in enumerate(map_type):
        lane = map_xy[indx]
        if type == 0:
            continue
        if type in [1, 2, 3]:
            # 使用灰色虚线表示中心线
            color = 'grey'
            linestyle = 'dotted'
            linewidth = 1
        else:
            color = 'grey'
            linestyle = '-'
            linewidth = 0.2

        # 绘制线条
        for i in range(len(lane) - 1):
            if lane[i, 0] and lane[i + 1, 0]:
                ax.plot([lane[i, 0], lane[i + 1, 0]], [lane[i, 1], lane[i + 1, 1]],
                        linewidth=linewidth, color=color, linestyle=linestyle)

    # 设置坐标轴比例和范围
    vis_range = 35
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(True)
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    #plt.show()
    return ax

def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image


def visualize_prediction(batch, prediction, draw_index=0):
    batch = batch['input_dict']
    map_lanes = batch['map_polylines'][draw_index].cpu().numpy()
    map_mask = batch['map_polylines_mask'][draw_index].cpu().numpy()
    past_traj = batch['obj_trajs'][draw_index].cpu().numpy()
    past_traj_mask = batch['obj_trajs_mask'][draw_index].cpu().numpy()
    center_gt_trajs = batch['center_gt_trajs'][draw_index].cpu().numpy()
    center_gt_trajs_mask = batch['center_gt_trajs_mask'][draw_index].cpu().numpy()
    track_index_to_predict = int(batch['track_index_to_predict'][draw_index].detach().cpu().item())
    pred_future_prob = prediction['predicted_probability'][draw_index].detach().cpu().numpy()
    pred_future_traj = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()
    scenario_id = str(batch['scenario_id'][draw_index])
    object_id = str(batch['center_objects_id'][draw_index])

    map_xy = map_lanes[..., :2]
    map_type = map_lanes[..., 0, -20:]
    if track_index_to_predict < past_traj.shape[0] and past_traj_mask[track_index_to_predict].any():
        target_past_index = track_index_to_predict
    else:
        valid_counts = past_traj_mask.astype(bool).sum(axis=-1)
        target_past_index = int(np.argmax(valid_counts))

    target_past_xy = past_traj[target_past_index, :, :2]
    target_past_valid = past_traj_mask[target_past_index].astype(bool)
    target_future_xy = center_gt_trajs[:, :2]
    target_future_valid = center_gt_trajs_mask.astype(bool)

    if pred_future_traj.ndim == 3:
        top_mode = int(np.nanargmax(pred_future_prob))
        target_pred_xy = pred_future_traj[top_mode, :, :2]
        top_probability = float(pred_future_prob[top_mode])
    else:
        top_mode = 0
        target_pred_xy = pred_future_traj[:, :2]
        top_probability = float(pred_future_prob) if np.ndim(pred_future_prob) == 0 else float(np.nanmax(pred_future_prob))
    target_pred_xy = target_pred_xy[:, :2]

    pred_valid = np.isfinite(target_pred_xy).all(axis=-1)

    def valid_points(points, valid):
        if points.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        finite = np.isfinite(points).all(axis=-1)
        return points[valid & finite]

    past_points = valid_points(target_past_xy, target_past_valid)
    future_points = valid_points(target_future_xy, target_future_valid)
    pred_points = valid_points(target_pred_xy, pred_valid)
    focus_points = np.concatenate(
        [points for points in (past_points, future_points, pred_points) if len(points)],
        axis=0,
    ) if any(len(points) for points in (past_points, future_points, pred_points)) else np.zeros((1, 2), dtype=np.float32)

    x_min, y_min = np.nanmin(focus_points, axis=0)
    x_max, y_max = np.nanmax(focus_points, axis=0)
    padding = 15.0
    min_span = 35.0
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    span = max(float(x_max - x_min), float(y_max - y_min), min_span)
    half_span = span / 2.0 + padding
    xlim = (x_center - half_span, x_center + half_span)
    ylim = (y_center - half_span, y_center + half_span)

    def draw_polyline(points, valid, color, label, line_width=3.0, linestyle='-', marker='o', alpha=1.0, zorder=3):
        points = valid_points(points, valid)
        if len(points) == 0:
            return
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=line_width,
            linestyle=linestyle,
            marker=marker,
            markersize=3.0,
            markevery=max(len(points) // 8, 1),
            alpha=alpha,
            solid_capstyle='round',
            label=label,
            zorder=zorder,
        )
        ax.scatter(points[-1, 0], points[-1, 1], s=28, color=color, edgecolors='white', linewidths=0.6, zorder=zorder + 1)

    def in_view(points):
        if len(points) == 0:
            return False
        return (
            (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
            (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1])
        ).any()

    # draw map
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect('equal')
    for idx, lane in enumerate(map_xy):
        lane_valid = map_mask[idx].astype(bool)
        lane_points = valid_points(lane, lane_valid)
        if len(lane_points) < 2 or not in_view(lane_points):
            continue
        lane_type = int(np.argmax(map_type[idx]))
        line_width = 1.0 if lane_type in [1, 2, 3] else 0.7
        linestyle = ':' if lane_type in [1, 2, 3] else '-'
        ax.plot(
            lane_points[:, 0],
            lane_points[:, 1],
            color='#9a9a9a',
            linewidth=line_width,
            linestyle=linestyle,
            alpha=0.45,
            zorder=1,
        )

    draw_polyline(target_past_xy, target_past_valid, color='#333333', label='Past', line_width=2.0, linestyle='--', marker='.', alpha=0.85, zorder=4)
    draw_polyline(target_future_xy, target_future_valid, color='#1f77b4', label='GT future', line_width=3.0, marker='o', alpha=0.95, zorder=5)
    draw_polyline(target_pred_xy, pred_valid, color='#ff7f0e', label=f'Pred top-1 ({top_probability:.2f})', line_width=3.0, marker='o', alpha=0.95, zorder=6)

    if len(past_points):
        ax.scatter(
            past_points[-1, 0],
            past_points[-1, 1],
            s=52,
            color='black',
            edgecolors='white',
            linewidths=0.8,
            label='Current',
            zorder=8,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=9)
    ax.set_title(
        f'{scenario_id} | object {object_id} | top mode {top_mode}',
        fontsize=9,
        pad=8,
    )
    fig.tight_layout(pad=0.2)

    return plt


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rotate_xy(points, heading):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotated = np.empty_like(points[..., :2], dtype=np.float32)
    rotated[..., 0] = points[..., 0] * cos_h - points[..., 1] * sin_h
    rotated[..., 1] = points[..., 0] * sin_h + points[..., 1] * cos_h
    return rotated


def _local_to_world(points, center_object_world, map_center):
    points = np.asarray(points, dtype=np.float32)
    map_center = np.asarray(map_center, dtype=np.float32).reshape(-1)[:2]
    origin = np.asarray(center_object_world[:2], dtype=np.float32) + map_center
    return _rotate_xy(points[..., :2], float(center_object_world[6])) + origin


def _source_to_world(points, map_center):
    points = np.asarray(points, dtype=np.float32)
    map_center = np.asarray(map_center, dtype=np.float32).reshape(-1)[:2]
    return points[:, :2] + map_center


def extract_prediction_visualization_record(batch, prediction, draw_index=0):
    input_dict = batch["input_dict"]
    scenario_id = str(input_dict["scenario_id"][draw_index])
    object_id = str(input_dict["center_objects_id"][draw_index])
    object_type = None
    if "center_objects_type" in input_dict:
        object_type_value = input_dict["center_objects_type"][draw_index]
        object_type_array = _to_numpy(object_type_value)
        object_type = object_type_array.item() if np.ndim(object_type_array) == 0 else object_type_array.tolist()

    center_gt_src = _to_numpy(input_dict["center_gt_trajs_src"][draw_index])
    center_object_world = _to_numpy(input_dict["center_objects_world"][draw_index])
    map_center = _to_numpy(input_dict["map_center"][draw_index])

    if center_gt_src.ndim != 2 or center_gt_src.shape[0] < 81:
        return {
            "scenario_id": scenario_id,
            "object_id": object_id,
            "past": np.zeros((0, 2), dtype=np.float32),
            "gt": np.zeros((0, 2), dtype=np.float32),
            "pred": np.zeros((0, 2), dtype=np.float32),
            "object_type": object_type,
            "top_mode": -1,
            "top_probability": 0.0,
            "past_valid_count": 0,
            "gt_valid_count": 0,
            "pred_valid_count": 0,
            "is_target_track": False,
        }

    src_valid = center_gt_src[:, -1].astype(bool) & np.isfinite(center_gt_src[:, :2]).all(axis=-1)
    past_src = center_gt_src[:21, :2]
    gt_src = center_gt_src[21:81, :2]
    past_valid = src_valid[:21]
    gt_valid = src_valid[21:81]

    pred_prob = _to_numpy(prediction["predicted_probability"][draw_index])
    pred_traj = _to_numpy(prediction["predicted_trajectory"][draw_index])
    if pred_traj.ndim == 3:
        top_mode = int(np.nanargmax(pred_prob))
        pred_xy = pred_traj[top_mode, :, :2]
        top_probability = float(pred_prob[top_mode])
    else:
        top_mode = 0
        pred_xy = pred_traj[:, :2]
        top_probability = float(pred_prob) if np.ndim(pred_prob) == 0 else float(np.nanmax(pred_prob))

    pred_valid = np.isfinite(pred_xy).all(axis=-1)

    return {
        "scenario_id": scenario_id,
        "object_id": object_id,
        "past": _source_to_world(past_src[past_valid], map_center),
        "gt": _source_to_world(gt_src[gt_valid], map_center),
        "pred": _local_to_world(pred_xy[pred_valid], center_object_world, map_center),
        "object_type": object_type,
        "top_mode": top_mode,
        "top_probability": top_probability,
        "past_valid_count": int(past_valid.sum()),
        "gt_valid_count": int(gt_valid.sum()),
        "pred_valid_count": int(pred_valid.sum()),
        "is_target_track": True,
    }


def _feature_geometry(feature):
    if "polyline" in feature:
        geometry = np.asarray(feature.get("polyline", []), dtype=np.float32)
        closed = False
    elif "polygon" in feature:
        geometry = np.asarray(feature.get("polygon", []), dtype=np.float32)
        closed = True
    else:
        return np.zeros((0, 2), dtype=np.float32), False
    if geometry.ndim != 2 or geometry.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), closed
    return geometry[:, :2], closed


def prediction_record_path_length(record):
    return len(record.get("past", ())) + len(record.get("gt", ()))


def prediction_record_pred_path_length(record):
    return len(record.get("past", ())) + len(record.get("pred", ()))


def _record_xy(record, key):
    points = np.asarray(record.get(key, ()), dtype=np.float32)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = points[:, :2]
    return points[np.isfinite(points).all(axis=-1)]


def _path_displacement(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1, :2] - points[0, :2]))


def prediction_record_pred_displacement(record):
    return _path_displacement(_record_xy(record, "pred"))


def prediction_record_gt_displacement(record):
    return _path_displacement(_record_xy(record, "gt"))


def is_target_vehicle_prediction_record(record):
    object_type = record.get("object_type", None)
    if object_type is None:
        return bool(record.get("is_target_track", False))
    if isinstance(object_type, (np.integer, int)):
        return int(object_type) == 1
    normalized = str(object_type).strip().lower()
    return normalized in {"1", "vehicle", "type_vehicle", "car", "truck", "bus"} or normalized.endswith(".vehicle")


def is_drawable_prediction_record(record, min_total_steps=61):
    return (
        bool(record.get("is_target_track", False))
        and is_target_vehicle_prediction_record(record)
        and int(record.get("past_valid_count", 0)) == 21
        and int(record.get("gt_valid_count", 0)) == 60
        and int(record.get("pred_valid_count", 0)) >= 60
        and prediction_record_path_length(record) == 81
        and prediction_record_pred_path_length(record) >= min_total_steps
    )


def is_moving_prediction_record(record, min_total_steps=61, min_displacement=2.0):
    return (
        is_drawable_prediction_record(record, min_total_steps=min_total_steps)
        and prediction_record_gt_displacement(record) >= min_displacement
        and prediction_record_pred_displacement(record) >= min_displacement
    )


def prediction_record_diagnostics(records, min_total_steps=61, min_displacement=2.0):
    return {
        "candidates": len(records),
        "target_track": sum(1 for record in records if bool(record.get("is_target_track", False))),
        "target_vehicle": sum(1 for record in records if is_target_vehicle_prediction_record(record)),
        "past_21": sum(1 for record in records if int(record.get("past_valid_count", 0)) == 21),
        "gt_60": sum(1 for record in records if int(record.get("gt_valid_count", 0)) == 60),
        "pred_60": sum(1 for record in records if int(record.get("pred_valid_count", 0)) >= 60),
        "drawable": sum(1 for record in records if is_drawable_prediction_record(record, min_total_steps=min_total_steps)),
        "moving": sum(
            1
            for record in records
            if is_moving_prediction_record(record, min_total_steps=min_total_steps, min_displacement=min_displacement)
        ),
    }


def _resample_polyline(points, num_samples=32):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:, :2], num_samples, axis=0)

    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.repeat(points[:1, :2], num_samples, axis=0)

    target = np.linspace(0.0, total, num_samples)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def _trajectory_overlap_distance(path_a, path_b):
    samples_a = _resample_polyline(path_a)
    samples_b = _resample_polyline(path_b)
    if len(samples_a) == 0 or len(samples_b) == 0:
        return np.inf
    forward = np.linalg.norm(samples_a - samples_b, axis=-1).mean()
    reverse = np.linalg.norm(samples_a - samples_b[::-1], axis=-1).mean()
    return float(min(forward, reverse))


def _record_display_path(record):
    points = [_record_xy(record, key) for key in ("past", "gt")]
    points = [point for point in points if len(point)]
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(points, axis=0)


def _select_moving_non_overlapping_records(records, max_tracks, min_track_distance=4.0, min_total_steps=61, min_displacement=2.0):
    moving_records = [
        record
        for record in records
        if is_moving_prediction_record(record, min_total_steps=min_total_steps, min_displacement=min_displacement)
    ]
    ranked = sorted(
        moving_records,
        key=lambda item: (
            -prediction_record_pred_displacement(item),
            -prediction_record_gt_displacement(item),
            -float(item.get("top_probability", 0.0)),
            str(item.get("scenario_id", "")),
            str(item.get("object_id", "")),
        ),
    )
    selected = []
    selected_paths = []
    for record in ranked:
        path = _record_display_path(record)
        if all(_trajectory_overlap_distance(path, selected_path) >= min_track_distance for selected_path in selected_paths):
            selected.append(record)
            selected_paths.append(path)
        if len(selected) >= max_tracks:
            break
    return selected


def select_prediction_records_for_osm_map(
    records,
    max_tracks=8,
    min_track_distance=4.0,
    min_total_steps=61,
    min_displacement=2.0,
):
    return _select_moving_non_overlapping_records(
        records,
        max_tracks=max_tracks,
        min_track_distance=min_track_distance,
        min_total_steps=min_total_steps,
        min_displacement=min_displacement,
    )


def _plot_map_feature(ax, feature, xlim=None, ylim=None):
    geometry, closed = _feature_geometry(feature)
    if len(geometry) < 2:
        return
    if xlim is not None and ylim is not None:
        gx_min, gy_min = np.nanmin(geometry, axis=0)
        gx_max, gy_max = np.nanmax(geometry, axis=0)
        if gx_max < xlim[0] or gx_min > xlim[1] or gy_max < ylim[0] or gy_min > ylim[1]:
            return

    feature_type = str(feature.get("type", ""))
    if "BOUNDARY" in feature_type:
        style = {"color": "#111111", "linewidth": 1.2, "linestyle": "-"}
    elif "BROKEN" in feature_type:
        style = {"color": "#b8b8b8", "linewidth": 0.8, "linestyle": (0, (4, 4))}
    elif "ROAD_LINE" in feature_type:
        style = {"color": "#b0b0b0", "linewidth": 0.9, "linestyle": "-"}
    elif "LANE" in feature_type:
        style = {"color": "#9b9b9b", "linewidth": 0.55, "linestyle": ":"}
    elif feature_type == "CROSSWALK":
        style = {"color": "#c0c0c0", "linewidth": 0.8, "linestyle": "-"}
    else:
        style = {"color": "#c7c7c7", "linewidth": 0.6, "linestyle": "-"}

    plot_points = geometry
    if closed and len(geometry) >= 3:
        plot_points = np.vstack([geometry, geometry[:1]])
    ax.plot(plot_points[:, 0], plot_points[:, 1], alpha=0.95, zorder=1, **style)


def visualize_prediction_records_on_osm_map(
    map_features,
    records,
    max_tracks=8,
    min_track_distance=4.0,
    min_total_steps=61,
    min_displacement=2.0,
    title="Prediction trajectories",
):
    selected_records = select_prediction_records_for_osm_map(
        records,
        max_tracks=max_tracks,
        min_track_distance=min_track_distance,
        min_total_steps=min_total_steps,
        min_displacement=min_displacement,
    )
    if not selected_records:
        raise ValueError(f"No moving drawable target-vehicle prediction records for {title}")
    drawable_count = sum(1 for record in records if is_drawable_prediction_record(record, min_total_steps=min_total_steps))
    moving_count = sum(
        1
        for record in records
        if is_moving_prediction_record(record, min_total_steps=min_total_steps, min_displacement=min_displacement)
    )
    fig, ax = plt.subplots(figsize=(11, 9))

    focus_points = []
    for record in selected_records:
        for key in ("past", "gt", "pred"):
            if len(record.get(key, ())):
                focus_points.append(np.asarray(record[key], dtype=np.float32)[:, :2])
    if focus_points:
        stacked_focus = np.concatenate(focus_points, axis=0)
        x_min, y_min = np.nanmin(stacked_focus, axis=0)
        x_max, y_max = np.nanmax(stacked_focus, axis=0)
        padding = 14.0
        xlim = (float(x_min - padding), float(x_max + padding))
        ylim = (float(y_min - padding), float(y_max + padding))
    else:
        xlim = ylim = None

    for feature in map_features.values():
        _plot_map_feature(ax, feature, xlim=xlim, ylim=ylim)

    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    label_flags = {"past": False, "gt": False, "pred": False, "current": False}
    for record in selected_records:
        past = record["past"]
        gt = record["gt"]
        pred = record["pred"]
        if len(past):
            ax.plot(
                past[:, 0],
                past[:, 1],
                color="#777777",
                linewidth=1.5,
                linestyle="--",
                alpha=0.82,
                label="Past" if not label_flags["past"] else None,
                zorder=4,
            )
            ax.scatter(
                past[-1, 0],
                past[-1, 1],
                s=24,
                color="#111111",
                edgecolors="white",
                linewidths=0.5,
                label="Current" if not label_flags["current"] else None,
                zorder=8,
            )
            label_flags["past"] = True
            label_flags["current"] = True
        if len(gt):
            ax.plot(
                gt[:, 0],
                gt[:, 1],
                color="#1f77b4",
                linewidth=2.5,
                marker="o",
                markersize=2.5,
                markevery=max(len(gt) // 8, 1),
                alpha=0.9,
                solid_capstyle="round",
                label="GT future" if not label_flags["gt"] else None,
                zorder=5,
            )
            label_flags["gt"] = True
        if len(pred):
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                color="#e4572e",
                linewidth=2.5,
                marker="o",
                markersize=2.5,
                markevery=max(len(pred) // 8, 1),
                alpha=0.9,
                solid_capstyle="round",
                label="Pred top-1" if not label_flags["pred"] else None,
                zorder=6,
            )
            label_flags["pred"] = True

    ax.set_title(
        f"{title} | moving non-overlap {len(selected_records)}/{moving_count} moving ({drawable_count} drawable, {len(records)} candidates)",
        fontsize=12,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout(pad=0.2)
    return plt
