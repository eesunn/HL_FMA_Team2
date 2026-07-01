# Segment-based motion primitive for differential drive execution.
#
# ROTATE   : 제자리에서 target_yaw (absolute, odom frame) 까지 회전.
# STRAIGHT : 현재 위치에서 distance (m, +전진/-후진) 만큼 직진. target_yaw는 무시.

uint8 TYPE_ROTATE   = 0
uint8 TYPE_STRAIGHT = 1

uint8 type
float64 target_yaw   # rad (ROTATE only). odom frame absolute.
float64 distance     # m   (STRAIGHT only). +forward, -backward.
