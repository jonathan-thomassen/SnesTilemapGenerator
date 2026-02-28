"""parse_tilemap.py — SNES tilemap hex → Tiled asset pipeline.

Three operating modes, selected automatically from the arguments:

  pipeline  parse a single SNES tilemap .txt file together with its
            tileset PNG and produce a Tiled TMX/TSX pair.

            python parse_tilemap.py map.txt tileset.png

  stitch    concatenate two or more SNES tilemap .txt files side-by-side
            to form a wider multi-screen map (128, 192, … columns), then
            run the same pipeline on the combined data.

            python parse_tilemap.py map1.txt map2.txt tileset.png

  clean     remove unused tiles from an existing TMX/TSX/PNG triple
            in-place, compacting the tileset and remapping all GIDs.

            python parse_tilemap.py map.tmx

SNES tilemap format
-------------------
Each map cell is a 16-bit little-endian word:
  low byte         — low 8 bits of tile index
  high byte bit 0  — bit 8 of tile index (+0x100)
  high byte bits 2-5 — palette index (0-15)
  high byte bit 6  — horizontal flip
  high byte bit 7  — vertical flip
All tile indices have a base offset of 0x200 applied on read.

SNES screens are 32x32 tiles.  Multi-screen maps store each screen
sequentially; ``build_array_2d`` interleaves them back into a single
continuous row-major grid.
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from defusedxml import ElementTree
from PIL import Image

TILE_SIZE = 8

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TileEntry:
    """One cell in an SNES tilemap.

    Attributes:
        index:   Zero-based tile index within the tileset image.
        hflip:   Horizontal flip flag.
        vflip:   Vertical flip flag.
        palette: SNES palette index (0-15 raw; re-interpreted as a
                 palette-index / color-group pair by
                 ``apply_palette_offsets``).

    """

    index: int
    hflip: bool
    vflip: bool
    palette: int = 0


@dataclass
class PaletteInfo:
    """Indexed-color palette data for a Pillow ``'P'``-mode image.

    Attributes:
        data:         Flat RGB bytes for every palette entry, padded to
                      256 entries (768 bytes) as required by Pillow's
                      ``putpalette``.
        transparency: Transparency index (int) or alpha bytes object to
                      pass directly to Pillow / PNG.  ``None`` if the
                      image has no transparency.
        num_colors:   Number of meaningful palette entries (e.g. 4 for a
                      2bpp tileset, 16 for a 4bpp one).  Used to derive
                      the PNG bit depth when saving.

    """

    data: bytes
    transparency: int | bytes | None = None
    num_colors: int | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_hex_file(filename: str) -> list[int]:
    """Read whitespace-separated hexadecimal byte tokens from *filename*.

    Each token must be a valid base-16 integer (e.g. ``0x3A`` or ``3A``).
    Empty lines and leading/trailing whitespace are ignored.

    Returns:
        A flat list of integer byte values in file order.

    """
    values: list[int] = []
    with Path.open(filename) as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            values.extend(int(token, 16) for token in line_stripped.split())
    return values


def parse_entries(values: list[int]) -> list[TileEntry]:
    """Convert raw byte pairs into TileEntry objects.

    Each 16-bit word is stored little-endian (low byte first).
    High byte layout:
      bit 0 (0x01)    → bit 8 of tile index (add 0x100)
      bits 2-5 (0x3C) → palette index (0-15)
      bit 6 (0x40)    → horizontal flip
      bit 7 (0x80)    → vertical flip
    Base offset of 0x200 is added to all tile indices.
    """
    return [
        TileEntry(
            index=values[i] + 0x200 + (0x100 if values[i + 1] & 0x01 else 0),
            hflip=bool(values[i + 1] & 0x40),
            vflip=bool(values[i + 1] & 0x80),
            palette=(values[i + 1] >> 2) & 0x0F,
        )
        for i in range(0, len(values), 2)
    ]


# ---------------------------------------------------------------------------
# 2-D array helpers
# ---------------------------------------------------------------------------


def build_array_2d(
    entries: list[TileEntry],
    cols: int,
    screen_cols: int = 32,
) -> list[list[TileEntry]]:
    """Build a 2-D grid from a flat entry list.

    SNES tilemap data is stored screen-by-screen: all rows of screen 0 come
    first, then all rows of screen 1, etc.  When *cols* is a multiple of
    *screen_cols* (default 32), the screens are interleaved row-by-row so that
    screen 0 occupies the left *screen_cols* columns, screen 1 the next, etc.
    For a single screen (cols == screen_cols) the simple row-major order is
    used unchanged.
    """
    total = len(entries)
    rows = total // cols

    if cols <= screen_cols or cols % screen_cols != 0:
        # Single screen or non-standard layout: simple row-major order.
        return [entries[r * cols : (r + 1) * cols] for r in range(rows)]

    # Multiple screens tiled horizontally: interleave one row per screen.
    num_screens = cols // screen_cols
    screen_size = rows * screen_cols  # entries per screen
    result: list[list[TileEntry]] = []
    for r in range(rows):
        row: list[TileEntry] = []
        for s in range(num_screens):
            start = s * screen_size + r * screen_cols
            row.extend(entries[start : start + screen_cols])
        result.append(row)
    return result


def log_array_2d(array_2d: list[list[TileEntry]]) -> None:
    """Emit a formatted dump of *array_2d* at DEBUG log level.

    Each cell is rendered as ``0xINDEXH V P<pal>`` where ``H``/``V``
    are shown only when the corresponding flip flag is set.  The output
    is emitted as a single ``log.debug`` call so it is suppressed unless
    ``--verbose`` is passed.
    """
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows else 0
    lines = [
        f"\n2D array  ({rows} rows x {cols} columns):",
        "(H = h-flip, V = v-flip, P = palette)",
        "[",
    ]
    for r, row in enumerate(array_2d):
        cell_strs = [
            f"0x{e.index:02X}{'H' if e.hflip else ' '}"
            f"{'V' if e.vflip else ' '}P{e.palette}"
            for e in row
        ]
        comma = "," if r < rows - 1 else ""
        lines.append(f"  [{', '.join(cell_strs)}]{comma}")
    lines.append("]")
    log.debug("%s", "\n".join(lines))


# ---------------------------------------------------------------------------
# Output: XML & CSV
# ---------------------------------------------------------------------------


def save_xml(array_2d: list[list[TileEntry]], out_path: Path) -> None:
    """Serialise *array_2d* to an XML file at *out_path*.

    The root element is ``<tilemap rows=… cols=…>``.  Each row becomes a
    ``<row index=…>`` element containing ``<tile hflip=… vflip=…
    palette=…>0xINDEX</tile>`` children.
    """
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows else 0
    root = ElementTree.Element("tilemap", rows=str(rows), cols=str(cols))
    for r, row in enumerate(array_2d):
        row_el = ElementTree.SubElement(root, "row", index=str(r))
        for e in row:
            tile_el = ElementTree.SubElement(
                row_el,
                "tile",
                hflip="1" if e.hflip else "0",
                vflip="1" if e.vflip else "0",
                palette=str(e.palette),
            )
            tile_el.text = f"0x{e.index:02X}"
    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(
        out_path,
        encoding="utf-8",
        xml_declaration=True,
    )
    log.info("Saved to %s", out_path)


def save_csv(array_2d: list[list[TileEntry]], stem_path: Path) -> None:
    """Write per-cell data from *array_2d* to four CSV files.

    Files written (derived from *stem_path*):
      * ``<stem>.csv``          — tile indices (hex)
      * ``<stem>_hflip.csv``   — horizontal-flip flags (0/1)
      * ``<stem>_vflip.csv``   — vertical-flip flags (0/1)
      * ``<stem>_palette.csv`` — palette indices
    """
    idx_path = stem_path.with_suffix(".csv")
    hflip_path = stem_path.with_name(stem_path.stem + "_hflip.csv")
    vflip_path = stem_path.with_name(stem_path.stem + "_vflip.csv")
    palette_path = stem_path.with_name(stem_path.stem + "_palette.csv")

    with Path.open(idx_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([f"0x{e.index:02X}" for e in row])

    with Path.open(hflip_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([int(e.hflip) for e in row])

    with Path.open(vflip_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([int(e.vflip) for e in row])

    with Path.open(palette_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([e.palette for e in row])

    log.info("Saved to %s, %s, %s, %s", idx_path, hflip_path, vflip_path, palette_path)


# ---------------------------------------------------------------------------
# Tileset loading & deduplication
# ---------------------------------------------------------------------------


def load_tiles(png_path: Path) -> tuple[list[Image.Image], PaletteInfo | None, int]:
    """Slice a tileset PNG into TILE_SIZExTILE_SIZE images.

    If the source image is in Indexed Color Mode ("P"), the palette and any
    transparency index are preserved in a PaletteInfo so output images can be
    saved in the same mode with full transparency retained.  For all other
    modes the image is converted to RGBA.
    Returns (tiles, palette, color_count) where palette is None for non-indexed
    images and color_count is the number of distinct colors used in the source PNG.
    """
    tileset = Image.open(png_path)
    if tileset.mode == "P":
        pal_data = bytes(tileset.getpalette() or [])
        pal_transparency = tileset.info.get("transparency")
        color_count: int = len(set(tileset.getdata()))
        palette: PaletteInfo | None = PaletteInfo(
            pal_data,
            pal_transparency,
            color_count,
        )
        trans_note = (
            f", transparency index {pal_transparency}"
            if pal_transparency is not None
            else ""
        )
        log.info(
            "Tileset mode: Indexed Color (P) — palette preserved%s, %d colors",
            trans_note,
            color_count,
        )
    else:
        tileset = tileset.convert("RGBA")
        palette = None
        color_count = len(set(tileset.getdata()))
        log.info("Tileset mode: %s", tileset.mode)

    ts_w, ts_h = tileset.size

    if ts_w % TILE_SIZE != 0 or ts_h % TILE_SIZE != 0:
        log.exception(
            "Tileset image size (%dx%d) is not a multiple of %dpx in both dimensions.",
            ts_w,
            ts_h,
            TILE_SIZE,
        )
        sys.exit(1)

    tiles_per_row = ts_w // TILE_SIZE
    tiles_per_col = ts_h // TILE_SIZE
    log.info(
        "Tileset: %dx%d px → %d tiles (%d per row)",
        ts_w,
        ts_h,
        tiles_per_row * tiles_per_col,
        tiles_per_row,
    )

    tiles: list[Image.Image] = []
    for ty in range(tiles_per_col):
        for tx in range(tiles_per_row):
            x0, y0 = tx * TILE_SIZE, ty * TILE_SIZE
            tiles.append(tileset.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)))
    return tiles, palette, color_count


def deduplicate_tiles(
    tiles: list[Image.Image],
    array_2d: list[list[TileEntry]],
) -> list[Image.Image]:
    """Remove duplicate tiles (including h/v/hv-flipped variants).

    Updates array_2d entries in-place to point to canonical indices
    and adjusts their hflip/vflip flags accordingly.
    Returns the deduplicated tile list.
    """
    unique_tiles: list[Image.Image] = []
    # Maps raw pixel bytes of any flip variant → (canonical_idx, hflip, vflip).
    # Every new unique tile registers all four variants at once, so lookups
    # are O(1) and no inner loop is required.
    variant_lookup: dict[bytes, tuple[int, bool, bool]] = {}
    remap: dict[int, tuple[int, bool, bool]] = {}

    for orig_idx, tile in enumerate(tiles):
        normal = tile.tobytes()
        match = variant_lookup.get(normal)
        if match is not None:
            remap[orig_idx] = match
        else:
            new_idx = len(unique_tiles)
            remap[orig_idx] = (new_idx, False, False)
            unique_tiles.append(tile)
            hflipped = tile.transpose(Image.FLIP_LEFT_RIGHT).tobytes()
            vflipped = tile.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
            hvflipped = (
                tile.transpose(Image.FLIP_LEFT_RIGHT)
                .transpose(Image.FLIP_TOP_BOTTOM)
                .tobytes()
            )
            variant_lookup[normal] = (new_idx, False, False)
            variant_lookup[hflipped] = (new_idx, True, False)
            variant_lookup[vflipped] = (new_idx, False, True)
            variant_lookup[hvflipped] = (new_idx, True, True)

    removed = len(tiles) - len(unique_tiles)
    log.info(
        "Deduplication: %d duplicates removed, %d unique tiles remain",
        removed,
        len(unique_tiles),
    )

    # XOR existing flags with the required correction flags.
    for row in array_2d:
        for e in row:
            new_idx, needs_hflip, needs_vflip = remap[e.index]
            e.index = new_idx
            e.hflip = e.hflip ^ needs_hflip
            e.vflip = e.vflip ^ needs_vflip

    return unique_tiles


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------


def validate_indices(array_2d: list[list[TileEntry]], tile_count: int) -> None:
    """Verify every tile index in *array_2d* is within ``[0, tile_count)``.

    Logs an error for each out-of-range cell and calls ``sys.exit(1)``
    if any are found.
    """
    bad = [
        (r, c, e.index)
        for r, row in enumerate(array_2d)
        for c, e in enumerate(row)
        if e.index >= tile_count
    ]
    if bad:
        for r, c, idx in bad:
            log.exception(
                "Index 0x%02X (%d) at [%d][%d] is out of range "
                "(tileset has %d tiles, max index %d).",
                idx,
                idx,
                r,
                c,
                tile_count,
                tile_count - 1,
            )
        sys.exit(1)


def remove_unused_tiles(
    tiles: list[Image.Image],
    array_2d: list[list[TileEntry]],
) -> list[Image.Image]:
    """Remove tiles that are not referenced by any entry in array_2d.

    Updates array_2d entries in-place to point to the new compacted indices.
    Returns the compacted tile list.
    """
    used: set[int] = {e.index for row in array_2d for e in row}
    # Build ordered list of used indices (preserves original order)
    kept = sorted(used)
    removed = len(tiles) - len(kept)
    log.info(
        "Unused tile removal: %d unused tiles removed, %d tiles remain",
        removed,
        len(kept),
    )

    # Map old index → new compact index
    remap = {old: new for new, old in enumerate(kept)}

    for row in array_2d:
        for e in row:
            e.index = remap[e.index]

    return [tiles[i] for i in kept]


# 16-color canonical sub-palette table for 4-color SNES tilesets.
# Sub-palette P occupies indices P*4 … P*4+3.
# Stored as 256-entry RGB flat bytes (768 bytes) as Pillow requires.
_SNES_GRAYSCALE_16: bytes = bytes(
    [
        0x00,
        0x00,
        0x00,  # 0  - group 0
        0x30,
        0x30,
        0x30,  # 1
        0x40,
        0x40,
        0x40,  # 2
        0x50,
        0x50,
        0x50,  # 3
        0x60,
        0x60,
        0x60,  # 4  - group 1
        0x70,
        0x70,
        0x70,  # 5
        0x80,
        0x80,
        0x80,  # 6
        0x90,
        0x90,
        0x90,  # 7
        0x98,
        0x98,
        0x98,  # 8  - group 2
        0xA0,
        0xA0,
        0xA0,  # 9
        0xAA,
        0xAA,
        0xAA,  # 10
        0xBB,
        0xBB,
        0xBB,  # 11
        0xCC,
        0xCC,
        0xCC,  # 12 - group 3
        0xDD,
        0xDD,
        0xDD,  # 13
        0xEE,
        0xEE,
        0xEE,  # 14
        0xFF,
        0xFF,
        0xFF,  # 15
    ],
) + bytes(240 * 3)  # pad to 256 entries


def apply_palette_offsets(
    tiles: list[Image.Image],
    array_2d: list[list[TileEntry]],
    palette: PaletteInfo,
    stride: int = 4,
) -> tuple[list[Image.Image], PaletteInfo, int]:
    """Create per-(tile, palette-group) variants with remapped pixel indices.

    For a tileset with *stride* colors per sub-palette, each tile used with
    palette group P has its pixel indices shifted by ``P * stride`` so that
    they land in the correct slot of the canonical 16-color grayscale table
    (_SNES_GRAYSCALE_16).  SNES palette indices are compacted to a dense
    0, 1, 2, … range so that only the groups actually referenced contribute
    to the color count.  Tiles used with multiple palette groups are
    duplicated.

    *array_2d* is updated in-place.  Returns the expanded tile list, a new
    PaletteInfo built from the canonical table, and the total color slot count.
    """
    # Decompose each SNES palette into:
    #   PI  (palette index, output to TMX) = snes_pal // stride
    #   CG  (color group, drives pixel offset) = snes_pal % stride
    # Tiles with the same (tile_index, CG) are visually identical regardless
    # of PI, so we deduplicate on (tile_index, CG).
    used_cg_pairs: set[tuple[int, int]] = {
        (e.index, e.palette % stride) for row in array_2d for e in row
    }
    sorted_cg_pairs = sorted(used_cg_pairs)
    variant_map: dict[tuple[int, int], int] = {
        pair: new_idx for new_idx, pair in enumerate(sorted_cg_pairs)
    }

    # Build one tile variant per (tile_index, CG) with pixel indices offset
    # by CG * stride so they point into the correct slot of _SNES_GRAYSCALE_16.
    new_tiles: list[Image.Image] = []
    for old_idx, cg in sorted_cg_pairs:
        offset = cg * stride
        if offset == 0:
            t = tiles[old_idx].copy()
        else:
            raw = tiles[old_idx].tobytes()
            remapped = bytes(0 if px == 0 else px + offset for px in raw)
            t = Image.frombytes("P", tiles[old_idx].size, remapped)
        t.putpalette(_SNES_GRAYSCALE_16)
        new_tiles.append(t)

    # Update array_2d: tile index → variant, e.palette → PI = snes_pal // stride.
    for row in array_2d:
        for e in row:
            cg = e.palette % stride
            e.index = variant_map[(e.index, cg)]
            e.palette = e.palette // stride - 2

    num_groups = len(sorted_cg_pairs)

    total_slots = num_groups * stride
    log.info(
        "Palette expansion (stride=%d): %d variant(s) from %d unique tile(s) "
        "across %d sub-palette(s) \u2192 %d color slots",
        stride,
        len(new_tiles),
        len(tiles),
        num_groups,
        total_slots,
    )

    canonical_palette = PaletteInfo(
        data=_SNES_GRAYSCALE_16,
        transparency=palette.transparency,
        num_colors=16,
    )
    return new_tiles, canonical_palette, total_slots


def _new_image(
    mode: str,
    size: tuple[int, int],
    palette: PaletteInfo | None,
) -> Image.Image:
    """Create a blank output image, restoring the palette if mode is 'P'."""
    img = Image.new(mode, size)
    if mode == "P" and palette is not None:
        img.putpalette(palette.data)
    return img


def _save_image(img: Image.Image, out_path: Path, palette: PaletteInfo | None) -> None:
    """Save *img* to *out_path*, injecting transparency metadata when present.

    For indexed images whose PaletteInfo carries num_colors, the PNG bit depth
    is derived automatically (e.g. 16 colors → 4bpp) to avoid bloating the
    palette chunk with unused black entries.
    """
    kwargs: dict = {}
    if palette is not None:
        if palette.transparency is not None:
            kwargs["transparency"] = palette.transparency
        if palette.num_colors is not None and palette.num_colors > 0:
            bits = max(1, (palette.num_colors - 1).bit_length())
            # Pillow PNG encoder accepts bits=1/2/4/8 for P-mode images.
            if bits in (1, 2, 4, 8):
                kwargs["bits"] = bits
    img.save(out_path, **kwargs)


def save_tileset(
    tiles: list[Image.Image],
    out_path: Path,
    palette: PaletteInfo | None,
    tiles_per_row: int = 16,
) -> None:
    """Pack *tiles* into a grid PNG and save it to *out_path*.

    Tiles are arranged left-to-right, top-to-bottom with up to
    *tiles_per_row* columns.  The image mode and bit depth are derived
    from *palette*: ``'P'`` (indexed) when a palette is provided,
    ``'RGBA'`` otherwise.  Transparency and bit-depth hints from
    *palette* are forwarded to the PNG encoder via ``_save_image``.
    """
    mode = "P" if palette is not None else "RGBA"
    cols = min(tiles_per_row, len(tiles))
    rows = -(-len(tiles) // cols)  # ceiling division
    img = _new_image(mode, (cols * TILE_SIZE, rows * TILE_SIZE), palette)
    for idx, tile in enumerate(tiles):
        tx = idx % cols
        ty = idx // cols
        img.paste(tile, (tx * TILE_SIZE, ty * TILE_SIZE))
    _save_image(img, out_path, palette)
    log.info(
        "Tileset image (%d tiles, %dx%d px) saved to %s",
        len(tiles),
        cols * TILE_SIZE,
        rows * TILE_SIZE,
        out_path,
    )


def render_image(
    array_2d: list[list[TileEntry]],
    tiles: list[Image.Image],
    out_path: Path,
    palette: PaletteInfo | None,
) -> None:
    """Compose the full tilemap into a single PNG and save it to *out_path*.

    Each cell in *array_2d* is looked up in *tiles* and pasted at the
    correct pixel position, with flip transforms applied.  The output
    image uses mode ``'P'`` when *palette* is provided, otherwise
    ``'RGBA'``.
    """
    mode = "P" if palette is not None else "RGBA"
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows else 0
    out_img = _new_image(mode, (cols * TILE_SIZE, rows * TILE_SIZE), palette)
    for r, row in enumerate(array_2d):
        for c, e in enumerate(row):
            tile = tiles[e.index]
            if e.hflip:
                tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
            if e.vflip:
                tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
            out_img.paste(tile, (c * TILE_SIZE, r * TILE_SIZE))
    _save_image(out_img, out_path, palette)
    log.info(
        "Output image (%dx%d px) saved to %s",
        cols * TILE_SIZE,
        rows * TILE_SIZE,
        out_path,
    )


# ---------------------------------------------------------------------------
# Tiled export (TMX + TSX)
# ---------------------------------------------------------------------------

# Tiled encodes flip flags in the upper bits of each 32-bit GID.
_TILED_HFLIP: int = 0x80000000
_TILED_VFLIP: int = 0x40000000


def save_tiled(
    array_2d: list[list[TileEntry]],
    tile_count: int,
    stem_path: Path,
    tiles_per_row: int = 16,
) -> None:
    """Write a Tiled-compatible TSX tileset and TMX tilemap.

    GIDs are 1-based (0 = empty).  Flip flags are OR-ed into the high bits
    of each GID as Tiled expects.
    """
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows else 0
    name = stem_path.stem

    ts_cols = min(tiles_per_row, tile_count)
    ts_rows = -(-tile_count // ts_cols)  # ceiling division
    tileset_png_name = name + "_tileset.png"
    tsx_path = stem_path.with_name(name + ".tsx")
    tmx_path = stem_path.with_suffix(".tmx")

    # --- TSX ---
    tsx_root = ElementTree.Element(
        "tileset",
        version="1.10",
        name=name,
        tilewidth=str(TILE_SIZE),
        tileheight=str(TILE_SIZE),
        tilecount=str(tile_count),
        columns=str(ts_cols),
    )
    ElementTree.SubElement(
        tsx_root,
        "image",
        source=tileset_png_name,
        width=str(ts_cols * TILE_SIZE),
        height=str(ts_rows * TILE_SIZE),
    )
    ElementTree.indent(tsx_root, space="  ")
    ElementTree.ElementTree(tsx_root).write(
        tsx_path,
        encoding="UTF-8",
        xml_declaration=True,
    )
    log.info("Saved Tiled tileset to %s", tsx_path)

    # --- TMX ---
    # Build CSV data: trailing comma on every row except the last (Tiled format).
    csv_rows: list[str] = []
    for r, row in enumerate(array_2d):
        gids: list[str] = []
        for e in row:
            gid = e.index + 1  # Tiled GIDs are 1-based
            if e.hflip:
                gid |= _TILED_HFLIP
            if e.vflip:
                gid |= _TILED_VFLIP
            gids.append(str(gid))
        row_str = ",".join(gids)
        csv_rows.append(row_str + ("," if r < rows - 1 else ""))

    csv_data = "\n" + "\n".join(csv_rows) + "\n"

    # --- Palette CSV layer ---
    pal_rows: list[str] = []
    for r, row in enumerate(array_2d):
        pal_values = [str(e.palette) for e in row]
        pal_rows.append(",".join(pal_values) + ("," if r < rows - 1 else ""))
    pal_data = "\n" + "\n".join(pal_rows) + "\n"

    # Write TMX directly as a string to avoid ElementTree.indent corrupting the CSV.
    tmx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<map version="1.10" tiledversion="1.10.2" orientation="orthogonal"'
        f' renderorder="right-down"'
        f' width="{cols}" height="{rows}"'
        f' tilewidth="{TILE_SIZE}" tileheight="{TILE_SIZE}"'
        f' infinite="0" nextlayerid="3" nextobjectid="1">',
        f'  <tileset firstgid="1" source="{tsx_path.name}"/>',
        f'  <layer id="1" name="Tiles" width="{cols}" height="{rows}">',
        f'    <data encoding="csv">{csv_data}    </data>',
        "  </layer>",
        f'  <layer id="2" name="Palette" width="{cols}" height="{rows}">',
        f'    <data encoding="csv">{pal_data}    </data>',
        "  </layer>",
        "</map>",
    ]
    tmx_path.write_text("\n".join(tmx_lines), encoding="UTF-8")
    log.info("Saved Tiled map to %s", tmx_path)


# Tiled stores flip flags in the upper 3 bits of each GID.
_TILED_GID_MASK: int = 0x1FFFFFFF


# ---------------------------------------------------------------------------
# TMX clean mode
# ---------------------------------------------------------------------------


def parse_tmx_gids(tmx_path: Path) -> tuple[ElementTree.Element, list[int], int]:
    """Parse *tmx_path* and extract the tile-layer GID list.

    Reads the first ``<tileset>`` and ``<layer>`` elements; the layer
    must use CSV encoding.  Exits with an error message if either
    element is missing or the encoding is unsupported.

    Returns:
        A 3-tuple ``(map_root, flat_gids, firstgid)`` where:

        * ``map_root``  — the parsed ``<map>`` ``Element`` (full tree).
        * ``flat_gids`` — flat list of raw 32-bit GID values in row-major
          order (flip bits preserved in the upper bits).
        * ``firstgid``  — the ``firstgid`` attribute of the tileset element.

    """
    tree = ElementTree.parse(tmx_path)
    root = tree.getroot()

    tileset_el = root.find("tileset")
    if tileset_el is None:
        log.exception("No <tileset> element found in TMX.")
        sys.exit(1)
    firstgid = int(tileset_el.get("firstgid", "1"))

    layer = root.find("layer")
    if layer is None:
        log.exception("No <layer> element found in TMX.")
        sys.exit(1)
    data_el = layer.find("data")
    if data_el is None or data_el.get("encoding") != "csv":
        log.exception("Only CSV-encoded TMX layers are supported.")
        sys.exit(1)

    raw = (data_el.text or "").strip()
    gids = [int(v) for v in raw.replace("\n", "").split(",") if v.strip()]
    return root, gids, firstgid


def _load_tsx(
    tmx_dir: Path,
    tileset_el: ElementTree.Element,
) -> tuple[
    ElementTree.ElementTree,
    ElementTree.Element,
    Path,
    int,
    int,
    ElementTree.Element,
    Path,
]:
    """Load and validate the TSX file referenced by *tileset_el*.

    Returns ``(tsx_tree, tsx_root_el, tsx_path, columns, tile_count,
    image_el, img_path)``.  Exits on any missing file or element.
    """
    tsx_src = tileset_el.get("source", "")
    tsx_path = (tmx_dir / tsx_src).resolve()
    if not tsx_path.exists():
        log.exception("TSX file '%s' not found.", tsx_path)
        sys.exit(1)

    tsx_tree = ElementTree.parse(tsx_path)
    tsx_root_el = tsx_tree.getroot()
    columns = int(tsx_root_el.get("columns", "16"))
    tile_count = int(tsx_root_el.get("tilecount", "0"))

    image_el = tsx_root_el.find("image")
    if image_el is None:
        log.exception("No <image> element found in TSX.")
        sys.exit(1)
    img_src = image_el.get("source", "")
    img_path = (tsx_path.parent / img_src).resolve()
    if not img_path.exists():
        log.exception("Tileset image '%s' not found.", img_path)
        sys.exit(1)
    return tsx_tree, tsx_root_el, tsx_path, columns, tile_count, image_el, img_path


def _slice_tileset_image(
    img_path: Path,
) -> tuple[list[Image.Image], PaletteInfo | None, str]:
    """Open *img_path*, extract palette info, and slice into TILE_SIZE tiles.

    Returns ``(tiles, palette, src_mode)`` where *palette* is ``None`` for
    non-indexed images and *src_mode* is the original Pillow image mode.
    """
    tileset_img = Image.open(img_path)
    src_mode = tileset_img.mode
    palette: PaletteInfo | None = None
    if src_mode == "P":
        palette = PaletteInfo(
            data=bytes(tileset_img.getpalette() or []),
            transparency=tileset_img.info.get("transparency"),
        )
    ts_w, ts_h = tileset_img.size
    if ts_w % TILE_SIZE != 0 or ts_h % TILE_SIZE != 0:
        log.exception(
            "Tileset image size (%dx%d) is not a multiple of %dpx.",
            ts_w,
            ts_h,
            TILE_SIZE,
        )
        sys.exit(1)
    tiles: list[Image.Image] = []
    for ty in range(ts_h // TILE_SIZE):
        for tx in range(ts_w // TILE_SIZE):
            x0, y0 = tx * TILE_SIZE, ty * TILE_SIZE
            tiles.append(tileset_img.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)))
    return tiles, palette, src_mode


def _remap_gids(gids: list[int], remap: dict[int, int], firstgid: int) -> list[str]:
    """Return a flat list of remapped GID strings (flip flags preserved).

    Empty tiles (base index < 0) become ``"0"``.
    """
    result: list[str] = []
    for gid in gids:
        flags = gid & ~_TILED_GID_MASK
        base = (gid & _TILED_GID_MASK) - firstgid
        result.append("0" if base < 0 else str(flags | (remap[base] + firstgid)))
    return result


def _gids_to_csv(new_gids: list[str], map_width: int) -> str:
    """Format a flat GID string list into a Tiled-style CSV block.

    Each row ends with a comma except the last.  The block is wrapped
    in leading and trailing newlines as Tiled expects.
    """
    rows_out = len(new_gids) // map_width if map_width else 0
    csv_rows: list[str] = []
    for r in range(rows_out):
        row_gids = new_gids[r * map_width : (r + 1) * map_width]
        csv_rows.append(",".join(row_gids) + ("," if r < rows_out - 1 else ""))
    return "\n" + "\n".join(csv_rows) + "\n"


def _write_updated_tsx(
    tsx_tree: ElementTree.ElementTree,
    tsx_root_el: ElementTree.Element,
    image_el: ElementTree.Element,
    new_tile_count: int,
    new_ts_cols: int,
    new_ts_rows: int,
    tsx_path: Path,
) -> None:
    """Update tileset dimensions in *tsx_root_el* and write *tsx_path*."""
    tsx_root_el.set("tilecount", str(new_tile_count))
    tsx_root_el.set("columns", str(new_ts_cols))
    image_el.set("width", str(new_ts_cols * TILE_SIZE))
    image_el.set("height", str(new_ts_rows * TILE_SIZE))
    ElementTree.indent(tsx_root_el, space="  ")
    tsx_tree.write(tsx_path, encoding="UTF-8", xml_declaration=True)
    log.info("Updated TSX saved to %s", tsx_path)


def _serialize_extra_layers(
    tmx_root: ElementTree.Element,
    tile_layer_el: ElementTree.Element,
) -> list[str]:
    """Serialize all TMX children except ``<tileset>`` and *tile_layer_el*.

    CSV ``<layer>`` elements are re-formatted to canonical style;
    ``<objectgroup>`` elements are serialized verbatim with two-space
    indentation.  Returns a list of string lines for insertion into the
    TMX output.
    """
    lines: list[str] = []
    for child in tmx_root:
        if child.tag == "tileset" or child is tile_layer_el:
            continue
        if child.tag == "layer":
            _append_csv_layer(child, lines)
        elif child.tag == "objectgroup":
            ElementTree.indent(child, space="  ")
            og_str = ElementTree.tostring(child, encoding="unicode")
            lines.append("\n".join("  " + line for line in og_str.splitlines()))
    return lines


def _append_csv_layer(child: ElementTree.Element, lines: list[str]) -> None:
    """Append a re-formatted CSV ``<layer>`` element to *lines*."""
    data_sub = child.find("data")
    if data_sub is None or data_sub.get("encoding") != "csv":
        return
    child_attrs = " ".join(f'{k}="{v}"' for k, v in child.attrib.items())
    raw_rows = [
        row.strip().rstrip(",")
        for row in (data_sub.text or "").splitlines()
        if row.strip()
    ]
    child_csv = (
        "\n"
        + "\n".join(
            row + ("," if i < len(raw_rows) - 1 else "")
            for i, row in enumerate(raw_rows)
        )
        + "\n"
    )
    lines.append(f"  <layer {child_attrs}>")
    lines.append(f'    <data encoding="csv">{child_csv}    </data>')
    lines.append("  </layer>")


def clean_tmx(tmx_path: Path) -> None:
    """Remove unused tiles from the tileset referenced by a TMX file.

    Updates the tileset image, TSX, and TMX in-place.
    """
    tmx_dir = tmx_path.parent
    tmx_root, gids, firstgid = parse_tmx_gids(tmx_path)

    tileset_el = tmx_root.find("tileset")
    if tileset_el is None:
        msg = "No <tileset> element found after parsing."
        raise ValueError(msg)
    tsx_src = tileset_el.get("source", "")

    tsx_tree, tsx_root_el, tsx_path, columns, tile_count, image_el, img_path = (
        _load_tsx(tmx_dir, tileset_el)
    )
    log.info("TMX:      %s", tmx_path)
    log.info("TSX:      %s", tsx_path)
    log.info("Tileset:  %s  (%d tiles, %d columns)", img_path, tile_count, columns)

    used = {b for gid in gids if (b := (gid & _TILED_GID_MASK) - firstgid) >= 0}
    unused_count = tile_count - len(used)
    if unused_count == 0:
        log.info("No unused tiles found — nothing to do.")
        return
    log.info("Found %d unused tile(s) out of %d; removing...", unused_count, tile_count)

    tiles, palette, src_mode = _slice_tileset_image(img_path)

    kept = sorted(used)
    remap = {old: new for new, old in enumerate(kept)}
    new_tile_count = len(kept)

    layer_el = tmx_root.find("layer")
    if layer_el is None:
        msg = "No <layer> element found after parsing."
        raise ValueError(msg)
    map_width = int(layer_el.get("width", "0"))
    csv_text = _gids_to_csv(_remap_gids(gids, remap, firstgid), map_width)

    new_ts_cols = min(columns, new_tile_count)
    new_ts_rows = -(-new_tile_count // new_ts_cols)
    mode = "P" if palette is not None else src_mode
    new_img = _new_image(
        mode,
        (new_ts_cols * TILE_SIZE, new_ts_rows * TILE_SIZE),
        palette,
    )
    for new_idx, old_idx in enumerate(kept):
        tx, ty = new_idx % new_ts_cols, new_idx // new_ts_cols
        new_img.paste(tiles[old_idx], (tx * TILE_SIZE, ty * TILE_SIZE))
    _save_image(new_img, img_path, palette)
    log.info("Updated tileset image (%d tiles) saved to %s", new_tile_count, img_path)

    _write_updated_tsx(
        tsx_tree,
        tsx_root_el,
        image_el,
        new_tile_count,
        new_ts_cols,
        new_ts_rows,
        tsx_path,
    )

    map_attrs = " ".join(f'{k}="{v}"' for k, v in tmx_root.attrib.items())
    layer_attrs = " ".join(f'{k}="{v}"' for k, v in layer_el.attrib.items())
    tmx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<map {map_attrs}>",
        f'  <tileset firstgid="{firstgid}" source="{tsx_src}"/>',
        f"  <layer {layer_attrs}>",
        f'    <data encoding="csv">{csv_text}    </data>',
        "  </layer>",
        *_serialize_extra_layers(tmx_root, layer_el),
        "</map>",
    ]
    tmx_path.write_text("\n".join(tmx_lines), encoding="UTF-8")
    log.info("Updated TMX saved to %s", tmx_path)
    log.info("Done — removed %d unused tile(s).", unused_count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="parse_tilemap.py",
        description=(
            "Parse an SNES tilemap hex file or clean an existing TMX.\n\n"
            "Modes:\n"
            "  pipeline : parse_tilemap.py map.txt tileset.png\n"
            "  stitch   : parse_tilemap.py map1.txt [map2.txt ...] tileset.png\n"
            "  clean    : parse_tilemap.py map.tmx"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help=(
            "pipeline: <map.txt> <tileset.png>  |  "
            "stitch: <map1.txt> [map2.txt ...] <tileset.png>  |  "
            "clean: <map.tmx>"
        ),
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=int,
        default=None,
        metavar="N",
        help="Number of map columns (derived from total/rows by default)",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=None,
        metavar="N",
        help="Number of map rows per screen (default: 32); columns = total / rows",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write index/hflip/vflip/palette CSV files",
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Also write the full tilemap XML file",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render the composed map as a PNG image",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate tiles (including flipped variants)",
    )
    parser.add_argument(
        "--remove-unused",
        action="store_true",
        help="Remove tiles not referenced by any map cell",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging (includes 2-D array dump)",
    )
    return parser


def _parse_stitch_mode(
    files: list[str],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[list[TileEntry], int, Path, Path]:
    """Load and combine N .txt files for stitch mode.

    Returns ``(entries, cols, stem_path, png_path)``.
    """
    txt_files, png_path = files[:-1], Path(files[-1])
    if not png_path.exists():
        log.exception("PNG file '%s' not found.", png_path)
        sys.exit(1)
    try:
        all_values = [parse_hex_file(txt_file) for txt_file in txt_files]
    except FileNotFoundError:
        log.exception("Input file not found.")
        sys.exit(1)
    except ValueError:
        log.exception("Invalid hex value in input file.")
        sys.exit(1)
    values = [v for part in all_values for v in part]
    if len(values) % 2 != 0:
        log.exception(
            "Combined data has an odd number of bytes (%d); "
            "cannot form complete 16-bit words.",
            len(values),
        )
        sys.exit(1)
    entries = parse_entries(values)
    row_count = args.rows if args.rows is not None else 32
    if len(entries) % row_count != 0:
        parser.error(
            f"Stitched entry count {len(entries)} is not "
            f"divisible by {row_count} rows.",
        )
    cols = len(entries) // row_count
    counts_str = " + ".join(str(len(v) // 2) for v in all_values)
    log.info(
        "Total tile entries after stitch: %d (%s) \u2192 %d columns \u00d7 %d rows.",
        len(entries),
        counts_str,
        cols,
        row_count,
    )
    stems = [Path(f).stem for f in txt_files]
    stem_path = Path(txt_files[0]).with_name("_".join(stems) + "_stitched")
    return entries, cols, stem_path, png_path


def _parse_pipeline_mode(
    files: list[str],
) -> tuple[list[TileEntry], Path, Path]:
    """Load a single .txt file for pipeline mode.

    Returns ``(entries, stem_path, png_path)``.
    """
    filename, png_path = files[0], Path(files[1])
    if not png_path.exists():
        log.exception("PNG file '%s' not found.", png_path)
        sys.exit(1)
    try:
        values = parse_hex_file(filename)
    except FileNotFoundError:
        log.exception("File '%s' not found.", filename)
        sys.exit(1)
    except ValueError:
        log.exception("Invalid hex value in '%s'.", filename)
        sys.exit(1)
    if len(values) % 2 != 0:
        log.exception(
            "File has an odd number of bytes (%d); cannot form complete 16-bit words.",
            len(values),
        )
        sys.exit(1)
    entries = parse_entries(values)
    log.info("Total tile entries: %d", len(entries))
    return entries, Path(filename), png_path


def _cols_from_both(
    total: int,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Validate and return column count when both ``-c`` and ``-r`` are given."""
    if args.columns <= 0:
        parser.error("-c/--columns must be a positive integer")
    if args.rows <= 0:
        parser.error("-r/--rows must be a positive integer")
    if args.columns * args.rows != total:
        parser.error(
            f"-c {args.columns} \u00d7 -r {args.rows} = {args.columns * args.rows} "
            f"but file has {total} entries.",
        )
    log.info("Using %d columns \u00d7 %d rows.", args.columns, args.rows)
    return args.columns


def _cols_from_columns(
    total: int,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Validate and return column count when only ``-c`` is given."""
    if args.columns <= 0:
        parser.error("-c/--columns must be a positive integer")
    if total % args.columns != 0:
        valid = ", ".join(str(c) for c in range(1, total + 1) if total % c == 0)
        parser.error(
            f"{total} entries cannot be evenly divided into {args.columns} columns "
            f"(remainder: {total % args.columns}). Valid column counts: {valid}",
        )
    log.info("Using %d columns.", args.columns)
    return args.columns


def _cols_from_rows(
    total: int,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Derive column count from ``-r`` (default 32 rows)."""
    row_count = args.rows if args.rows is not None else 32
    if row_count <= 0:
        parser.error("-r/--rows must be a positive integer")
    if total % row_count != 0:
        valid = ", ".join(str(c) for c in range(1, total + 1) if total % c == 0)
        parser.error(
            f"{total} entries cannot be evenly divided into {row_count} rows "
            f"(remainder: {total % row_count}). Valid row counts: {valid}",
        )
    cols = total // row_count
    screen_info = (
        f" ({cols // 32} screen(s) of 32\u00d7{row_count})" if cols % 32 == 0 else ""
    )
    log.info("Using %d columns \u00d7 %d rows%s.", cols, row_count, screen_info)
    return cols


def _resolve_pipeline_cols(
    entries: list[TileEntry],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Choose the appropriate column-resolution strategy for pipeline mode."""
    total = len(entries)
    if args.columns is not None and args.rows is not None:
        return _cols_from_both(total, args, parser)
    if args.columns is not None:
        return _cols_from_columns(total, args, parser)
    return _cols_from_rows(total, args, parser)


def _run_tilemap_pipeline(
    array_2d: list[list[TileEntry]],
    stem_path: Path,
    png_path: Path,
    args: argparse.Namespace,
) -> None:
    """Load tiles, apply optional optimisations, then write all outputs."""
    if args.xml:
        save_xml(array_2d, stem_path.with_suffix(".xml"))
    if args.csv:
        save_csv(array_2d, stem_path)
    tiles, palette, color_count = load_tiles(png_path)
    if args.deduplicate:
        tiles = deduplicate_tiles(tiles, array_2d)
    if args.remove_unused:
        tiles = remove_unused_tiles(tiles, array_2d)
    if palette is not None and color_count == 4:
        tiles, palette, color_count = apply_palette_offsets(
            tiles,
            array_2d,
            palette,
            stride=color_count,
        )
    validate_indices(array_2d, len(tiles))
    save_tileset(tiles, stem_path.with_name(stem_path.stem + "_tileset.png"), palette)
    if args.render:
        render_image(array_2d, tiles, stem_path.with_suffix(".png"), palette)
    save_tiled(array_2d, len(tiles), stem_path)


def main() -> None:
    """CLI entry point — parse arguments and dispatch to the correct mode.

    Determines the operating mode from the positional file arguments:

    * **pipeline** — one ``.txt`` + one ``.png``: parse a single SNES
      tilemap and produce TMX/TSX/tileset PNG.
    * **stitch** — two-or-more ``.txt`` files + one ``.png``: concatenate
      multiple SNES tilemaps side-by-side before running the pipeline.
    * **clean** — single ``.tmx``: strip unused tiles from an existing
      Tiled asset set in-place.
    """
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    files = args.files
    if len(files) == 1 and files[0].lower().endswith(".tmx"):
        tmx_path = Path(files[0])
        if not tmx_path.exists():
            log.exception("File '%s' not found.", tmx_path)
            sys.exit(1)
        clean_tmx(tmx_path)
        return

    if len(files) >= 3 and files[-1].lower().endswith(".png"):
        entries, cols, stem_path, png_path = _parse_stitch_mode(files, args, parser)
    elif len(files) == 2 and files[1].lower().endswith(".png"):
        entries, stem_path, png_path = _parse_pipeline_mode(files)
        cols = _resolve_pipeline_cols(entries, args, parser)
    else:
        parser.error(
            "expected one of:\n"
            "  pipeline : <map.txt> <tileset.png>\n"
            "  stitch   : <map1.txt> [map2.txt ...] <tileset.png>\n"
            "  clean    : <map.tmx>",
        )

    array_2d = build_array_2d(entries, cols)
    log_array_2d(array_2d)
    _run_tilemap_pipeline(array_2d, stem_path, png_path, args)


if __name__ == "__main__":
    main()
