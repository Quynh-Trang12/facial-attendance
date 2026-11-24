"""Display-only identity text on the existing canvas; never an identity tracker."""
from dataclasses import dataclass
from .webcam_prototype import box_in_frame


@dataclass(frozen=True)
class IdentityLabel:
    text: str
    known: bool = False


def identity_label(state, view, *, face_count, confirmed, processing, trained, unverified_preview=False):
    """Use the current raw result or a guarded, unverified local display sample.

    The caller only passes visually continuous local outlines. Neither an
    overlay nor a retained outline can authorize any operation.
    """
    if not trained or face_count != 1 or state.ui_state in {
            'no_face', 'multiple_faces', 'spoof', 'uncertain', 'error'}:
        return None
    if unverified_preview:
        if view is not None and view.source_context is not None and (confirmed or view.historical):
            if view.identity_name:
                return IdentityLabel(' '.join(view.identity_name.split()), known=True)
            if view.identity == 'Unknown':
                return IdentityLabel('Unknown')
        return IdentityLabel('Checking match')
    if processing or state.ui_state == 'checking':
        return IdentityLabel('Checking...')
    result = state.recognition
    live = state.ui_state == 'live' and state.liveness is not None and state.liveness.status == 'live'
    if (not confirmed or state.context is None or not live
            or result is None or result.source != 'model'
            or view is None or view.historical or view.source_context != state.context):
        return None
    expected = f'{result.name} [{result.person_id}]' if result.accepted else 'Unknown'
    if view.identity != expected:
        return None
    if result.accepted:
        if not isinstance(result.name, str) or not result.name.strip():
            return None
        return IdentityLabel(' '.join(result.name.split()), known=True)
    return IdentityLabel('Unknown')


def preview_box(box, source_size, preview_size, *, mirrored=False):
    """Use the same top-left, half-open transform for outline and label."""
    width, height = source_size
    view_width, view_height = preview_size
    if min(width, height, view_width, view_height) <= 0 or not box_in_frame(box, width, height):
        raise ValueError('Invalid preview region')
    x1, y1, x2, y2 = box
    mapped = (x1*view_width/width, y1*view_height/height,
              x2*view_width/width, y2*view_height/height)
    if mirrored:
        return view_width-mapped[2], mapped[1], view_width-mapped[0], mapped[3]
    return mapped


def fit_text(label, maximum, measure):
    suffix = ' · UNVERIFIED' if label.known else ''
    text = label.text + suffix
    if measure(text) <= maximum:
        return text
    if measure('…' + suffix) > maximum:
        return None
    low, high = 0, len(label.text)
    while low < high:
        middle = (low+high+1)//2
        if measure(label.text[:middle]+'…'+suffix) <= maximum:
            low = middle
        else:
            high = middle-1
    return label.text[:low]+'…'+suffix


def label_bounds(box, preview_size, text_width, text_height):
    """Above first, below at a top edge, then a clamped inside-edge fallback."""
    width, height = preview_size
    margin, gap = 4, 3
    if text_width > width-2*margin or text_height > height-2*margin:
        return None
    left = max(margin, min(box[0], width-margin-text_width))
    if box[1]-gap-text_height >= margin:
        top = box[1]-gap-text_height
    elif box[3]+gap+text_height <= height-margin:
        top = box[3]+gap
    else:
        top = max(margin, min(box[1]+gap, height-margin-text_height))
    return left, top, left+text_width, top+text_height


class CanvasIdentityOverlay:
    """Two reusable canvas items, updated only when visible content changes."""
    TAG = 'identity-overlay'

    def __init__(self, canvas):
        from tkinter import font
        self.canvas = canvas
        self.font = font.Font(root=canvas, font='TkDefaultFont', size=12, weight='bold')
        self.items = None
        self.key = None
        self._text_key = None
        self._text_size = None

    def reset(self):
        """The owning app has cleared the canvas (Retry/camera failure)."""
        self.items = None
        self.key = None

    def hide(self):
        if self.items is not None and self.key is not None:
            for item in self.items:
                self.canvas.itemconfigure(item, state='hidden')
        self.key = None

    def update(self, label, box, size):
        if label is None:
            self.hide()
            return
        if self._text_key != (label, size[0]):
            text = fit_text(label, size[0]-20, self.font.measure)
            self._text_key = label, size[0]
            self._text_size = (text, self.font.measure(text)+12 if text else 0, self.font.metrics('linespace')+8)
        text, width, height = self._text_size
        bounds = label_bounds(box, size, width, height) if text else None
        if bounds is None:
            self.hide()
            return
        key = text, bounds
        if key == self.key:
            return
        if self.items is None:
            background = self.canvas.create_rectangle(0, 0, 0, 0, fill='#14202b', outline='#ffce65',
                                                       width=1, tags=self.TAG)
            foreground = self.canvas.create_text(0, 0, text='', anchor='nw', fill='#fff6cf',
                                                 font=self.font, tags=self.TAG)
            self.items = background, foreground
        background, foreground = self.items
        self.canvas.coords(background, *bounds)
        self.canvas.coords(foreground, bounds[0]+6, bounds[1]+4)
        self.canvas.itemconfigure(background, state='normal')
        self.canvas.itemconfigure(foreground, text=text, state='normal')
        self.canvas.tag_raise(self.TAG)
        self.key = key
