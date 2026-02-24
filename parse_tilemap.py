import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

TILE_SIZE = 8


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TileEntry:
    index: int
    hflip: bool
    vflip: bool
    palette: int = 0


@dataclass
class PaletteInfo:
    """Indexed-color palette data, including optional transparency index."""

    data: bytes
    transparency: int | bytes | None = None
    num_colors: int | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_hex_file(filename: str) -> list[int]:
    values: list[int] = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for token in line.split():
                values.append(int(token, 16))
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


def print_array_2d(array_2d: list[list[TileEntry]]) -> None:
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows else 0
    print(f"\n2D array  ({rows} rows × {cols} columns):")
    print("(H = h-flip, V = v-flip, P = palette)")
    print("[")
    for r, row in enumerate(array_2d):
        cell_strs = [
            f"0x{e.index:02X}{'H' if e.hflip else ' '}{'V' if e.vflip else ' '}P{e.palette}"
            for e in row
        ]
        comma = "," if r < rows - 1 else ""
        print(f"  [{', '.join(cell_strs)}]{comma}")
    print("]")


# ---------------------------------------------------------------------------
# Output: XML & CSV
# ---------------------------------------------------------------------------


def save_xml(array_2d: list[list[TileEntry]], out_path: Path) -> None:
    rows = len(array_2d)
    cols = len(array_2d[0]) if rows else 0
    root = ET.Element("tilemap", rows=str(rows), cols=str(cols))
    for r, row in enumerate(array_2d):
        row_el = ET.SubElement(root, "row", index=str(r))
        for e in row:
            tile_el = ET.SubElement(
                row_el,
                "tile",
                hflip="1" if e.hflip else "0",
                vflip="1" if e.vflip else "0",
                palette=str(e.palette),
            )
            tile_el.text = f"0x{e.index:02X}"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved to {out_path}")


def save_csv(array_2d: list[list[TileEntry]], stem_path: Path) -> None:
    idx_path = stem_path.with_suffix(".csv")
    hflip_path = stem_path.with_name(stem_path.stem + "_hflip.csv")
    vflip_path = stem_path.with_name(stem_path.stem + "_vflip.csv")
    palette_path = stem_path.with_name(stem_path.stem + "_palette.csv")

    with Path.open(idx_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([f"0x{e.index:02X}" for e in row])

    with open(hflip_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([int(e.hflip) for e in row])

    with open(vflip_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([int(e.vflip) for e in row])

    with open(palette_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in array_2d:
            w.writerow([e.palette for e in row])

    print(f"Saved to {idx_path}, {hflip_path}, {vflip_path}, {palette_path}")


# ---------------------------------------------------------------------------
# Tileset loading & deduplication
# ---------------------------------------------------------------------------


def load_tiles(png_path: Path) -> tuple[list[Image.Image], PaletteInfo | None, int]:
    """Slice a tileset PNG into TILE_SIZE×TILE_SIZE images.

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
        print(
            f"Tileset mode: Indexed Color (P) — palette preserved{trans_note}, {color_count} colors",
        )
    else:
        tileset = tileset.convert("RGBA")
        palette = None
        color_count = len(set(tileset.getdata()))
        print(f"Tileset mode: {tileset.mode}")

    ts_w, ts_h = tileset.size

    if ts_w % TILE_SIZE != 0 or ts_h % TILE_SIZE != 0:
        print(
            f"Error: Tileset image size ({ts_w}×{ts_h}) is not a multiple of "
            f"{TILE_SIZE}px in both dimensions.",
        )
        sys.exit(1)

    tiles_per_row = ts_w // TILE_SIZE
    tiles_per_col = ts_h // TILE_SIZE
    print(
        f"Tileset: {ts_w}×{ts_h} px → "
        f"{tiles_per_row * tiles_per_col} tiles ({tiles_per_row} per row)",
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
    # Per unique tile: cached bytes for all four flip variants
    unique_variants: list[tuple[bytes, bytes, bytes, bytes]] = []
    # remap[orig_idx] = (canonical_idx, needs_hflip, needs_vflip)
    remap: dict[int, tuple[int, bool, bool]] = {}

    for orig_idx, tile in enumerate(tiles):
        normal = tile.tobytes()
        hflipped = tile.transpose(Image.FLIP_LEFT_RIGHT).tobytes()
        vflipped = tile.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        hvflipped = (
            tile.transpose(Image.FLIP_LEFT_RIGHT)
            .transpose(Image.FLIP_TOP_BOTTOM)
            .tobytes()
        )

        matched = False
        for new_idx, (vn, vh, vv, vhv) in enumerate(unique_variants):
            if normal == vn:
                remap[orig_idx] = (new_idx, False, False)
                matched = True
                break
            if normal == vh:
                remap[orig_idx] = (new_idx, True, False)
                matched = True
                break
            if normal == vv:
                remap[orig_idx] = (new_idx, False, True)
                matched = True
                break
            if normal == vhv:
                remap[orig_idx] = (new_idx, True, True)
                matched = True
                break

        if not matched:
            remap[orig_idx] = (len(unique_tiles), False, False)
            unique_tiles.append(tile)
            unique_variants.append((normal, hflipped, vflipped, hvflipped))

    removed = len(tiles) - len(unique_tiles)
    print(
        f"Deduplication: {removed} duplicates removed, "
        f"{len(unique_tiles)} unique tiles remain",
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
    bad = [
        (r, c, e.index)
        for r, row in enumerate(array_2d)
        for c, e in enumerate(row)
        if e.index >= tile_count
    ]
    if bad:
        for r, c, idx in bad:
            print(
                f"Error: index 0x{idx:02X} ({idx}) at [{r}][{c}] is out of range "
                f"(tileset has {tile_count} tiles, max index {tile_count - 1}).",
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
    print(
        f"Unused tile removal: {removed} unused tiles removed, {len(kept)} tiles remain",
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
    print(
        f"Palette expansion (stride={stride}): {len(new_tiles)} variant(s) "
        f"from {len(tiles)} unique tile(s) across {num_groups} sub-palette(s) "
        f"\u2192 {total_slots} color slots",
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
    """Save the (deduplicated) tile list back out as a tileset PNG."""
    mode = "P" if palette is not None else "RGBA"
    cols = min(tiles_per_row, len(tiles))
    rows = -(-len(tiles) // cols)  # ceiling division
    img = _new_image(mode, (cols * TILE_SIZE, rows * TILE_SIZE), palette)
    for idx, tile in enumerate(tiles):
        tx = idx % cols
        ty = idx // cols
        img.paste(tile, (tx * TILE_SIZE, ty * TILE_SIZE))
    _save_image(img, out_path, palette)
    print(
        f"Tileset image ({len(tiles)} tiles, {cols * TILE_SIZE}×{rows * TILE_SIZE} px) saved to {out_path}",
    )


def render_image(
    array_2d: list[list[TileEntry]],
    tiles: list[Image.Image],
    out_path: Path,
    palette: PaletteInfo | None,
) -> None:
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
    print(
        f"Output image ({cols * TILE_SIZE}×{rows * TILE_SIZE} px) saved to {out_path}",
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
    color_count: int = 8,
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
    tsx_root = ET.Element(
        "tileset",
        version="1.10",
        name=name,
        tilewidth=str(TILE_SIZE),
        tileheight=str(TILE_SIZE),
        tilecount=str(tile_count),
        columns=str(ts_cols),
    )
    ET.SubElement(
        tsx_root,
        "image",
        source=tileset_png_name,
        width=str(ts_cols * TILE_SIZE),
        height=str(ts_rows * TILE_SIZE),
    )
    ET.indent(tsx_root, space="  ")
    ET.ElementTree(tsx_root).write(tsx_path, encoding="UTF-8", xml_declaration=True)
    print(f"Saved Tiled tileset to {tsx_path}")

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

    # --- Palette object layer ---
    # Each cell becomes a transparent rectangle object carrying a "palette"
    # custom property.  Object layers are the only Tiled construct that
    # supports arbitrary per-cell custom properties.
    obj_lines: list[str] = []
    obj_id = 1
    for r, row in enumerate(array_2d):
        for c, e in enumerate(row):
            x = c * TILE_SIZE
            y = r * TILE_SIZE
            obj_lines.append(
                f'    <object id="{obj_id}" x="{x}" y="{y}"'
                f' width="{TILE_SIZE}" height="{TILE_SIZE}">',
            )
            obj_lines.append("      <properties>")
            obj_lines.append(
                f'        <property name="palette" type="int" value="{e.palette}"/>',
            )
            obj_lines.append("      </properties>")
            obj_lines.append("    </object>")
            obj_id += 1
    next_object_id = obj_id

    # Write TMX directly as a string to avoid ET.indent corrupting the CSV text.
    tmx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<map version="1.10" tiledversion="1.10.2" orientation="orthogonal"'
        f' renderorder="right-down"'
        f' width="{cols}" height="{rows}"'
        f' tilewidth="{TILE_SIZE}" tileheight="{TILE_SIZE}"'
        f' infinite="0" nextlayerid="3" nextobjectid="{next_object_id}">',
        f'  <tileset firstgid="1" source="{tsx_path.name}"/>',
        f'  <layer id="1" name="Tiles" width="{cols}" height="{rows}">',
        f'    <data encoding="csv">{csv_data}    </data>',
        "  </layer>",
        '  <objectgroup id="2" name="Palette">',
        "    <properties>",
        f'      <property name="color_count" type="int" value="{color_count}"/>',
        "    </properties>",
        *obj_lines,
        "  </objectgroup>",
        "</map>",
    ]
    tmx_path.write_text("\n".join(tmx_lines), encoding="UTF-8")
    print(f"Saved Tiled map to {tmx_path}")


# Tiled stores flip flags in the upper 3 bits of each GID.
_TILED_GID_MASK: int = 0x1FFFFFFF


# ---------------------------------------------------------------------------
# TMX clean mode
# ---------------------------------------------------------------------------


def parse_tmx_gids(tmx_path: Path) -> tuple[ET.Element, ET.Element, list[int], int]:
    """Parse a TMX file and return (map_root, data_el, flat_gids, firstgid)."""
    tree = ET.parse(tmx_path)
    root = tree.getroot()

    tileset_el = root.find("tileset")
    if tileset_el is None:
        print("Error: No <tileset> element found in TMX.")
        sys.exit(1)
    firstgid = int(tileset_el.get("firstgid", "1"))

    layer = root.find("layer")
    if layer is None:
        print("Error: No <layer> element found in TMX.")
        sys.exit(1)
    data_el = layer.find("data")
    if data_el is None or data_el.get("encoding") != "csv":
        print("Error: Only CSV-encoded TMX layers are supported.")
        sys.exit(1)

    raw = (data_el.text or "").strip()
    gids = [int(v) for v in raw.replace("\n", "").split(",") if v.strip()]
    return root, data_el, gids, firstgid


def clean_tmx(tmx_path: Path) -> None:
    """Remove unused tiles from the tileset referenced by a TMX file.

    Updates the tileset image, TSX, and TMX in-place.
    """
    tmx_dir = tmx_path.parent

    # --- Load TMX ---
    tmx_root, data_el, gids, firstgid = parse_tmx_gids(tmx_path)

    tileset_el = tmx_root.find("tileset")
    assert tileset_el is not None
    tsx_src = tileset_el.get("source", "")
    tsx_path = (tmx_dir / tsx_src).resolve()
    if not tsx_path.exists():
        print(f"Error: TSX file '{tsx_path}' not found.")
        sys.exit(1)

    # --- Load TSX ---
    tsx_tree = ET.parse(tsx_path)
    tsx_root_el = tsx_tree.getroot()
    columns = int(tsx_root_el.get("columns", "16"))
    tile_count = int(tsx_root_el.get("tilecount", "0"))

    image_el = tsx_root_el.find("image")
    if image_el is None:
        print("Error: No <image> element found in TSX.")
        sys.exit(1)
    img_src = image_el.get("source", "")
    img_path = (tsx_path.parent / img_src).resolve()
    if not img_path.exists():
        print(f"Error: Tileset image '{img_path}' not found.")
        sys.exit(1)

    print(f"TMX:      {tmx_path}")
    print(f"TSX:      {tsx_path}")
    print(f"Tileset:  {img_path}  ({tile_count} tiles, {columns} columns)")

    # --- Find used 0-based tile indices ---
    used: set[int] = set()
    for gid in gids:
        base = (gid & _TILED_GID_MASK) - firstgid
        if base >= 0:  # negative means empty tile (gid=0)
            used.add(base)

    unused_count = tile_count - len(used)
    if unused_count == 0:
        print("No unused tiles found — nothing to do.")
        return
    print(f"Found {unused_count} unused tile(s) out of {tile_count}; removing...")

    # --- Load tileset image tiles ---
    tileset_img = Image.open(img_path)
    palette: PaletteInfo | None = None
    if tileset_img.mode == "P":
        palette = PaletteInfo(
            data=bytes(tileset_img.getpalette() or []),
            transparency=tileset_img.info.get("transparency"),
        )
    ts_w, ts_h = tileset_img.size
    if ts_w % TILE_SIZE != 0 or ts_h % TILE_SIZE != 0:
        print(
            f"Error: Tileset image size ({ts_w}x{ts_h}) is not a multiple of {TILE_SIZE}px.",
        )
        sys.exit(1)

    tiles_per_row = ts_w // TILE_SIZE
    tiles_per_col = ts_h // TILE_SIZE
    tiles: list[Image.Image] = []
    for ty in range(tiles_per_col):
        for tx in range(tiles_per_row):
            x0, y0 = tx * TILE_SIZE, ty * TILE_SIZE
            tiles.append(tileset_img.crop((x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)))

    # --- Build compact remap: old 0-based index -> new 0-based index ---
    kept = sorted(used)
    remap = {old: new for new, old in enumerate(kept)}
    new_tile_count = len(kept)

    # --- Remap GIDs in the layer data ---
    new_gids: list[str] = []
    for gid in gids:
        flags = gid & ~_TILED_GID_MASK
        base = (gid & _TILED_GID_MASK) - firstgid
        if base < 0:
            new_gids.append("0")
        else:
            new_gids.append(str(flags | (remap[base] + firstgid)))

    layer_el = tmx_root.find("layer")
    assert layer_el is not None
    map_width = int(layer_el.get("width", "0"))
    rows_out = len(new_gids) // map_width if map_width else 0
    csv_rows: list[str] = []
    for r in range(rows_out):
        row_gids = new_gids[r * map_width : (r + 1) * map_width]
        csv_rows.append(",".join(row_gids) + ("," if r < rows_out - 1 else ""))
    csv_text = "\n" + "\n".join(csv_rows) + "\n"

    # --- Save updated tileset image ---
    new_ts_cols = min(columns, new_tile_count)
    new_ts_rows = -(-new_tile_count // new_ts_cols)
    mode = "P" if palette is not None else tileset_img.mode
    new_img = _new_image(
        mode,
        (new_ts_cols * TILE_SIZE, new_ts_rows * TILE_SIZE),
        palette,
    )
    for new_idx, old_idx in enumerate(kept):
        tx = new_idx % new_ts_cols
        ty = new_idx // new_ts_cols
        new_img.paste(tiles[old_idx], (tx * TILE_SIZE, ty * TILE_SIZE))
    _save_image(new_img, img_path, palette)
    print(f"Updated tileset image ({new_tile_count} tiles) saved to {img_path}")

    # --- Update TSX ---
    tsx_root_el.set("tilecount", str(new_tile_count))
    tsx_root_el.set("columns", str(new_ts_cols))
    image_el.set("width", str(new_ts_cols * TILE_SIZE))
    image_el.set("height", str(new_ts_rows * TILE_SIZE))
    ET.indent(tsx_root_el, space="  ")
    tsx_tree.write(tsx_path, encoding="UTF-8", xml_declaration=True)
    print(f"Updated TSX saved to {tsx_path}")

    # --- Save updated TMX (written as a string to preserve CSV integrity) ---
    map_attrs = " ".join(f'{k}="{v}"' for k, v in tmx_root.attrib.items())
    layer_attrs = " ".join(f'{k}="{v}"' for k, v in layer_el.attrib.items())
    tmx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<map {map_attrs}>",
        f'  <tileset firstgid="{firstgid}" source="{tsx_src}"/>',
        f"  <layer {layer_attrs}>",
        f'    <data encoding="csv">{csv_text}    </data>',
        "  </layer>",
        "</map>",
    ]
    tmx_path.write_text("\n".join(tmx_lines), encoding="UTF-8")
    print(f"Updated TMX saved to {tmx_path}")

    print(f"Done — removed {unused_count} unused tile(s).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parse_tilemap.py",
        description="Parse an SNES tilemap hex file or clean an existing TMX.",
    )
    parser.add_argument(
        "input",
        help="Tilemap .txt file (pipeline mode) or .tmx file (clean mode)",
    )
    parser.add_argument(
        "tileset",
        nargs="?",
        help="Tileset PNG — required for pipeline mode",
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
    args = parser.parse_args()

    # TMX clean mode: single .tmx argument
    if args.input.lower().endswith(".tmx"):
        tmx_path = Path(args.input)
        if not tmx_path.exists():
            print(f"Error: File '{tmx_path}' not found.")
            sys.exit(1)
        clean_tmx(tmx_path)
        return

    if args.tileset is None:
        parser.error("a tileset PNG is required in pipeline mode")

    filename = args.input
    png_path = Path(args.tileset)

    if not png_path.exists():
        print(f"Error: PNG file '{png_path}' not found.")
        sys.exit(1)

    try:
        values = parse_hex_file(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid hex value in file — {e}")
        sys.exit(1)

    if len(values) % 2 != 0:
        print(
            f"Error: File has an odd number of bytes ({len(values)}); "
            "cannot form complete 16-bit words.",
        )
        sys.exit(1)

    entries = parse_entries(values)
    print(f"Total tile entries: {len(entries)}")

    total = len(entries)
    if args.columns is not None and args.rows is not None:
        # Both explicitly provided: validate they account for all entries.
        if args.columns <= 0:
            parser.error("-c/--columns must be a positive integer")
        if args.rows <= 0:
            parser.error("-r/--rows must be a positive integer")
        if args.columns * args.rows != total:
            parser.error(
                f"-c {args.columns} \u00d7 -r {args.rows} = {args.columns * args.rows} "
                f"but file has {total} entries.",
            )
        cols = args.columns
        print(f"Using {cols} columns \u00d7 {args.rows} rows.")
    elif args.columns is not None:
        # Explicit columns only: derive rows = total / cols.
        if args.columns <= 0:
            parser.error("-c/--columns must be a positive integer")
        if total % args.columns != 0:
            valid = ", ".join(str(c) for c in range(1, total + 1) if total % c == 0)
            parser.error(
                f"{total} entries cannot be evenly divided into {args.columns} columns "
                f"(remainder: {total % args.columns}). Valid column counts: {valid}",
            )
        cols = args.columns
        print(f"Using {cols} columns.")
    else:
        # Derive columns from --rows (default 32): screens are tiled horizontally.
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
        screens = cols // 32 if cols % 32 == 0 else cols
        screen_info = (
            f" ({screens} screen(s) of 32\u00d7{row_count})" if cols % 32 == 0 else ""
        )
        print(f"Using {cols} columns \u00d7 {row_count} rows{screen_info}.")

    array_2d = build_array_2d(entries, cols)
    print_array_2d(array_2d)

    stem_path = Path(filename)
    if args.xml:
        save_xml(array_2d, stem_path.with_suffix(".xml"))
    if args.csv:
        save_csv(array_2d, stem_path)

    tiles, palette, color_count = load_tiles(png_path)
    tiles = deduplicate_tiles(tiles, array_2d)
    tiles = remove_unused_tiles(tiles, array_2d)
    if palette is not None and color_count == 4:
        tiles, palette, color_count = apply_palette_offsets(
            tiles,
            array_2d,
            palette,
            stride=color_count,
        )
    validate_indices(array_2d, len(tiles))
    tileset_out_path = stem_path.with_name(stem_path.stem + "_tileset.png")
    save_tileset(tiles, tileset_out_path, palette)
    if args.render:
        render_image(array_2d, tiles, stem_path.with_suffix(".png"), palette)
    save_tiled(array_2d, len(tiles), stem_path, color_count)


if __name__ == "__main__":
    main()
