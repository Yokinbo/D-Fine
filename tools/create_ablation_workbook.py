"""Generate the paper-ready D-FINE ablation workbook without third-party packages."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ACCURACY_ROWS = [
    ("D-FINE", 0.9176, 0.8159, 0.8828, 0.8479, 0.5871, 0.5748),
    ("D-FINE + DSQC", 0.9291, 0.8203, 0.9030, 0.8596, 0.5827, 0.5967),
    ("D-FINE + RBA", 0.9234, 0.8217, 0.8909, 0.8547, 0.5812, 0.5702),
    ("D-FINE + DSQC + RBA", 0.9329, 0.8384, 0.9010, 0.8686, 0.5986, 0.5999),
]

EFFICIENCY_ROWS = [
    ("D-FINE", 19.468, 37.345, 22.1311, 45.1853, "三次延迟均值；FPS=1000/平均延迟"),
    ("D-FINE + DSQC", 19.516, 37.373, 21.6134, 46.2676, "三次延迟均值；FPS=1000/平均延迟"),
    ("D-FINE + RBA", 19.468, 37.345, 19.8404, 50.4022, "三次延迟均值；FPS=1000/平均延迟"),
    ("D-FINE + DSQC + RBA", 19.516, 37.373, 21.4611, 46.5959, "三次延迟均值；FPS=1000/平均延迟"),
]

RAW_ROWS = [
    ("D-FINE", "SD18", 512, 19.468, 37.345, 0.9201, 0.8324, 0.8727, 0.8521, 0.5844, 0.5677, 22.4158, 44.6113, "是", "正式均值"),
    ("D-FINE", "SD2026", 512, 19.468, 37.345, 0.9225, 0.8305, 0.8909, 0.8596, 0.5931, 0.5818, 20.6702, 48.3788, "是", "正式均值"),
    ("D-FINE", "SD3407", 512, 19.468, 37.345, 0.9101, 0.7849, 0.8848, 0.8319, 0.5839, 0.5749, 23.3073, 42.9050, "是", "正式均值"),
    ("D-FINE + DSQC", "SD18", 512, 19.516, 37.373, 0.9287, 0.8305, 0.8909, 0.8596, 0.5521, 0.5908, 22.2460, 44.9519, "是", "正式均值"),
    ("D-FINE + DSQC", "SD2026", 512, 19.516, 37.373, 0.9169, 0.8177, 0.8970, 0.8555, 0.5662, 0.5947, 20.9683, 47.6911, "是", "正式均值"),
    ("D-FINE + DSQC", "SD3407", 512, 19.516, 37.373, 0.9416, 0.8128, 0.9212, 0.8636, 0.6298, 0.6046, 21.6259, 46.2408, "是", "正式均值"),
    ("D-FINE + RBA", "SD73", 512, 19.468, 37.345, 0.9172, 0.8043, 0.8970, 0.8481, 0.5686, 0.5661, 19.9503, 50.1246, "是", "按用户指定纳入；效率已重新测量"),
    ("D-FINE + RBA", "SD59", 512, 19.468, 37.345, 0.9231, 0.8421, 0.8727, 0.8571, 0.5978, 0.5688, 19.3624, 51.6465, "是", "按用户指定纳入"),
    ("D-FINE + RBA", "SD2026", 512, 19.468, 37.345, 0.9299, 0.8187, 0.9030, 0.8588, 0.5771, 0.5757, 20.2085, 49.4841, "是", "按用户指定纳入"),
    ("D-FINE + RBA", "SD3407", 512, 19.468, 37.345, 0.9025, 0.8122, 0.8909, 0.8497, 0.5228, 0.5505, 21.8039, 45.8634, "否", "有效运行但未计入指定三次均值"),
    ("D-FINE + DSQC + RBA", "SD18", 512, 19.516, 37.373, 0.9484, 0.8636, 0.9212, 0.8915, 0.5773, 0.5917, 21.2050, 47.1587, "是", "正式均值"),
    ("D-FINE + DSQC + RBA", "newSD2026", 512, 19.516, 37.373, 0.9267, 0.8315, 0.8970, 0.8630, 0.6338, 0.6053, 21.2472, 47.0649, "是", "正式均值"),
    ("D-FINE + DSQC + RBA", "SD3407", 512, 19.516, 37.373, 0.9236, 0.8202, 0.8848, 0.8513, 0.5848, 0.6028, 21.9311, 45.5973, "是", "正式均值"),
    ("D-FINE + DSQC + RBA", "SD300", 512, 19.516, 37.373, 0.9216, 0.8033, 0.8909, 0.8448, 0.5776, 0.5582, 56.2190, 17.7876, "否", "有效运行但未计入指定三次均值"),
]


def col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell(row: int, col: int, value, style: int = 0) -> str:
    ref = f"{col_name(col)}{row}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{text}</t></is></c>'


def row_xml(number: int, values, styles=None, height=None) -> str:
    styles = styles or [0] * len(values)
    height_xml = f' ht="{height}" customHeight="1"' if height else ""
    cells = "".join(cell(number, i + 1, value, styles[i]) for i, value in enumerate(values))
    return f'<row r="{number}"{height_xml}>{cells}</row>'


def worksheet(rows, widths, merges=(), freeze_row=None) -> str:
    cols = "".join(
        f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(widths, 1)
    )
    pane = ""
    if freeze_row:
        pane = f'<pane ySplit="{freeze_row}" topLeftCell="A{freeze_row + 1}" activePane="bottomLeft" state="frozen"/>'
    merge_xml = ""
    if merges:
        merge_xml = f'<mergeCells count="{len(merges)}">' + "".join(
            f'<mergeCell ref="{item}"/>' for item in merges
        ) + "</mergeCells>"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetViews><sheetView workbookViewId="0">{pane}</sheetView></sheetViews>'
        f'<cols>{cols}</cols><sheetData>{"".join(rows)}</sheetData>{merge_xml}'
        '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
        '</worksheet>'
    )


def main_sheet() -> str:
    rows = [
        row_xml(1, ["D-FINE模型消融实验（验证集，三次实验均值）"], [1], 26),
        row_xml(3, ["模型", "AP50", "P@0.5", "R@0.5", "F1@0.5", "AP75", "mAP50:95"], [2] * 7, 24),
    ]
    maxima = [max(row[col] for row in ACCURACY_ROWS) for col in range(1, 7)]
    for number, data in enumerate(ACCURACY_ROWS, 4):
        styles = [4] + [5 if value == maxima[i] else 3 for i, value in enumerate(data[1:])]
        rows.append(row_xml(number, data, styles, 22))
    rows.extend([
        row_xml(9, ["注：精度均为三次验证结果算术均值；RBA使用SD2026、SD59、SD73。"], [7], 28),
        row_xml(11, ["效率指标（验证集测速均值）"], [1], 25),
        row_xml(13, ["模型", "Params(M)", "FLOPs(G)", "Latency(ms/image)", "FPS", "说明"], [2] * 6, 24),
    ])
    for number, data in enumerate(EFFICIENCY_ROWS, 14):
        styles = [4, 3, 3, 3, 3, 7]
        rows.append(row_xml(number, data, styles, 24))
    rows.append(row_xml(19, ["效率建议：Params/FLOPs可直接入论文；Latency/FPS须在同一台电脑空闲状态下统一复测后定稿。"], [7], 32))
    return worksheet(rows, [29, 14, 14, 14, 14, 14, 16], ["A1:G1", "A9:G9", "A11:F11", "A19:F19"], 3)


def raw_sheet() -> str:
    headers = ["模型", "随机种子", "Input", "Params(M)", "FLOPs(G)", "AP50", "P@0.5", "R@0.5", "F1@0.5", "AP75", "mAP50:95", "Latency(ms)", "FPS", "计入均值", "备注"]
    rows = [row_xml(1, headers, [2] * len(headers), 24)]
    for number, data in enumerate(RAW_ROWS, 2):
        styles = [4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 7]
        if data[13] == "否":
            styles = [6] * 14 + [7]
        rows.append(row_xml(number, data, styles, 22))
    return worksheet(rows, [27, 14, 10, 13, 12, 11, 11, 11, 11, 11, 14, 16, 12, 12, 34], freeze_row=1)


def notes_sheet() -> str:
    notes = [
        "计算与论文使用说明",
        "1. 主精度表表头按用户给出的论文样式排列：模型、AP50、P@0.5、R@0.5、F1@0.5、AP75、mAP50:95。",
        "2. 所有精度均为表中标记“计入均值=是”的三次结果算术平均，并四舍五入至4位小数。",
        "3. 平均FPS统一按1000/平均Latency计算，不直接平均每次FPS。",
        "4. RBA为训练期约束，不增加推理参数和FLOPs；SD73已重新测速为19.9503ms、50.1246FPS，不再属于异常结果。",
        "5. RBA主表按用户指定采用SD2026、SD59、SD73；SD3407保留在逐次结果中但未计入。",
        "6. DSQC+RBA主表采用SD18、newSD2026、SD3407；SD300保留在逐次结果中但未计入。",
        "7. 若这些三次结果是从更多有效运行中按精度筛选，投稿时应披露筛选规则；更严格的做法是各消融统一随机种子。",
        "8. 主文建议展示精度表和Params/FLOPs；Latency/FPS可在同机统一复测后放入主表或独立效率表。",
    ]
    rows = [row_xml(1, [notes[0]], [1], 26)]
    rows.extend(row_xml(i, [note], [7], 34) for i, note in enumerate(notes[1:], 3))
    return worksheet(rows, [120], ["A1:A1"])


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts>
  <fonts count="4">
    <font><sz val="11"/><name val="Microsoft YaHei"/></font>
    <font><b/><sz val="15"/><name val="Microsoft YaHei"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="12"/><name val="Microsoft YaHei"/></font>
    <font><b/><sz val="11"/><name val="Microsoft YaHei"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF339A8"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE2F0D9"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9D9D9"/></left><right style="thin"><color rgb="FFD9D9D9"/></right><top style="thin"><color rgb="FFD9D9D9"/></top><bottom style="thin"><color rgb="FFD9D9D9"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="2" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="164" fontId="3" fillId="3" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="4" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView/></bookViews>
  <sheets><sheet name="消融实验表" sheetId="1" r:id="rId1"/><sheet name="逐次验证结果" sheetId="2" r:id="rId2"/><sheet name="计算说明" sheetId="3" r:id="rId3"/></sheets>
</workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>Codex</dc:creator><dc:title>D-FINE DSQC RBA消融实验</dc:title><dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created></cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Excel Compatible</Application></Properties>'''
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", main_sheet())
        archive.writestr("xl/worksheets/sheet2.xml", raw_sheet())
        archive.writestr("xl/worksheets/sheet3.xml", notes_sheet())
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
