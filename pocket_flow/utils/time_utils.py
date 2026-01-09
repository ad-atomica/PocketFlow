from __future__ import annotations


def timewait(time_gap: float) -> str:
    d = time_gap // (24 * 3600)
    d_h = time_gap % (24 * 3600)
    h = d_h // 3600
    h_m = d_h % 3600
    m = h_m // 60
    s = h_m % 60
    if d > 0:
        out = f"{int(d)}d {int(h)}h {int(m)}m {round(s, 2)}s"
    elif h > 0:
        out = f"{int(h)}h {int(m)}m {round(s, 2)}s"
    elif m > 0:
        out = f"{int(m)}m {round(s, 2)}s"
    else:
        out = f"{round(s, 2)}s"
    return out
