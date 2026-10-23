//! Brand mark hooks remain empty in this modified build because `docs/brand/LICENSE-brand.txt`
//! requires forks to remove ArtCraft marks.

/// Retained for optional-brand callers; modified builds draw no ArtCraft mark.
pub fn paint_wordmark(_ui: &egui::Ui, _left_center: egui::Pos2, _height: f32, _dark: bool) -> Option<egui::Rect> {
    None
}
