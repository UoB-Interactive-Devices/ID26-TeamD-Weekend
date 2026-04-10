from windows_toasts import Toast, WindowsToaster

_TOASTER = WindowsToaster("ID26 TeamD Demo")


def show_windows_toast(title: str, message: str) -> tuple[bool, str]:
    try:
        toast = Toast()
        toast.text_fields = [title, message]
        _TOASTER.show_toast(toast)
        return True, "Windows notification sent."
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"Notification failed: {exc}"
