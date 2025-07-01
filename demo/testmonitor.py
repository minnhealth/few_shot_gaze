import wmi

def parse_edid(edid_data):
    # EDID is 128 bytes; byte 21 = width_cm, byte 22 = height_cm
    width_cm  = edid_data[21]
    height_cm = edid_data[22]
    return width_cm * 10, height_cm * 10  # mm

def get_monitor_size_mm():
    w = wmi.WMI(namespace='root\\WMI')
    for m in w.WmiMonitorDescriptorMethods():
        raw = m.WmiGetMonitorRawEEdidV1Block(0)[0]    # raw is a list of ints
        edid_bytes = bytes(raw)                      # directly make it bytes
        return parse_edid(edid_bytes)

    raise RuntimeError("Could not read monitor EDID")

w_mm, h_mm = get_monitor_size_mm()
# 24 in x 13.5 in (monitor)
# 13.5 in x 8 in (laptop)
print(f"Width: {w_mm} mm, Height: {h_mm} mm")