import platform

if platform.system() == "Windows":
    from windows_toasts import Toast, WindowsToaster

    _TOASTER = WindowsToaster("Demo")
else:
    Toast = None
    _TOASTER = None


def show_windows_toast(title: str, message: str) -> tuple[bool, str]:
    if platform.system() != "Windows" or _TOASTER is None or Toast is None:
        return False, "Windows notifications are only supported on Windows."

    try:
        toast = Toast()
        toast.text_fields = [title, message]
        _TOASTER.show_toast(toast)
        return True, "Windows notification sent."
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Notification failed: {exc}"
