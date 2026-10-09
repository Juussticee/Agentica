"""One-time seeding of knowledge/books/ with two small real documents.

Creates:
  1. orbital_kinetics_principles.docx  (python-docx)
  2. urban_beekeeping_field_guide.pdf  (pypdf, minimal text PDF)

These are test fixtures for spec step 2; replace with real books anytime —
ingest_books.py will pick up new files automatically on the next run.
"""

from pathlib import Path

from docx import Document

BOOKS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "books"


def make_docx(path: Path) -> None:
    doc = Document()
    doc.add_heading("Orbital Kinetics: Principles of Spacecraft Motion", 0)
    doc.add_paragraph("Second edition, prepared for training purposes.")

    doc.add_heading("Chapter 1: Basic Orbits", level=1)
    doc.add_paragraph(
        "A circular orbit is the simplest trajectory: the spacecraft moves at constant "
        "altitude with a speed determined solely by the central body's gravitational "
        "parameter and the orbital radius. For low Earth orbit at 400 km altitude, the "
        "orbital speed is approximately 7.67 kilometers per second and one revolution "
        "takes about 92.4 minutes."
    )

    doc.add_heading("Chapter 2: Maneuvers", level=1)
    doc.add_paragraph(
        "The Hohmann transfer is the minimum-energy two-impulse maneuver connecting two "
        "circular, coplanar orbits. Moving from low Earth orbit to geostationary transfer "
        "orbit requires roughly 2.4 kilometers per second of delta-v, and circularizing at "
        "geostationary altitude adds approximately 1.5 kilometers per second more."
    )
    doc.add_paragraph(
        "Geostationary orbit lies at an altitude of approximately 35,786 kilometers above "
        "the equator, where the orbital period matches Earth's rotation, so the spacecraft "
        "appears fixed over one longitude."
    )

    doc.add_heading("Chapter 3: The Rocket Equation", level=1)
    doc.add_paragraph(
        "The Tsiolkovsky rocket equation relates delta-v to exhaust velocity and mass "
        "ratio: delta-v equals exhaust velocity times the natural logarithm of the initial "
        "mass divided by final mass. It was first derived in 1903 and remains the "
        "foundation of mission design."
    )
    doc.save(path)
    print(f"wrote {path}")


def make_pdf(path: Path) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import (
        DictionaryObject,
        NameObject,
        NumberObject,
        RectangleObject,
        StreamObject,
    )

    lines = [
        ("Field Guide to Urban Beekeeping", 16, 720),
        ("A practical handbook for small-apiary managers.", 11, 700),
        ("", 11, 682),
        ("Chapter 1: Colony Structure", 13, 664),
        ("A healthy hive in midsummer can contain up to 60,000 worker bees,", 11, 646),
        ("plus one queen and a few hundred drones depending on the season.", 11, 630),
        ("The queen lays up to 2,000 eggs per day at the peak of spring buildup.", 11, 614),
        ("", 11, 596),
        ("Chapter 2: Equipment", 13, 578),
        ("The Langstroth hive uses removable frames and relies on the bee space:", 11, 560),
        ("a corridor of 6 to 9 millimeters that bees keep free of comb and propolis.", 11, 544),
        ("Smokers calm bees by masking alarm pheromones during inspections.", 11, 528),
        ("", 11, 510),
        ("Chapter 3: Hive Products", 13, 492),
        ("Propolis is a resinous substance bees collect from plant buds and use", 11, 474),
        ("to seal small gaps and sterilize the interior of the hive.", 11, 458),
        ("Honey is harvested only when its moisture content drops below 18 percent,", 11, 442),
        ("which prevents fermentation during storage.", 11, 426),
    ]

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]
    page[NameObject("/MediaBox")] = RectangleObject([0, 0, 612, 792])

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )

    parts = ["BT"]
    for text, size, y in lines:
        if not text:
            continue
        safe = text.replace("(", " ").replace(")", " ")
        parts.append(f"/F1 {size} Tf 1 0 0 1 60 {y} Tm ({safe}) Tj")
    parts.append("ET")
    content = StreamObject()
    content.set_data(" ".join(parts).encode("latin-1", "replace"))
    stream_ref = writer._add_object(content)
    page[NameObject("/Contents")] = stream_ref
    page[NameObject("/Rotate")] = NumberObject(0)

    with open(path, "wb") as fh:
        writer.write(fh)
    print(f"wrote {path}")


if __name__ == "__main__":
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    make_docx(BOOKS_DIR / "orbital_kinetics_principles.docx")
    make_pdf(BOOKS_DIR / "urban_beekeeping_field_guide.pdf")
    print("seed complete")
