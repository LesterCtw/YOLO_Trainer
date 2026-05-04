from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_documentation_foundation_is_present() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    adr_dir = ROOT / "docs" / "adr"

    for required_text in [
        "Current status",
        "MVS direction",
        "macOS development",
        "Windows training workstation",
        "uv run yolo-trainer",
        "uv run pytest",
    ]:
        assert required_text in readme

    for domain_term in [
        "STEM ZC Image",
        "Metal Detection Box",
        "YOLO Training Project",
        "Normalized Training Image",
        "Reviewed Empty Image",
    ]:
        assert domain_term in context

    expected_adrs = [
        "0001-pyside6-desktop-gui.md",
        "0002-pretrained-yolo-fine-tuning.md",
        "0003-fixed-8-bit-normalization.md",
        "0004-windows-pip-venv-cuda-setup.md",
    ]
    for adr_name in expected_adrs:
        adr = adr_dir / adr_name
        assert adr.exists()
        assert "# ADR" in adr.read_text(encoding="utf-8")
