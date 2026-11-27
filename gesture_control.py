"""Entry point kept from v2 so existing commands still work:  python gesture_control.py --dry
Same as:  python -m gesture_hci --dry   (logic lives in the gesture_hci package)."""
from gesture_hci.app import main
from gesture_hci.controller import MODES, Controller  # noqa: F401  (v2 import compatibility)
from gesture_hci.geometry import classify_static, detect_swipe, finger_states, palm_still  # noqa: F401
from gesture_hci.kiosk import KioskController, OneEuro  # noqa: F401

if __name__ == "__main__":
    main()
