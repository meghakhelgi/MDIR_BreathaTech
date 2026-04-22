from __future__ import annotations

import json
import threading
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from conversion import (
    CURRENT_TO_PEROXIDE_INTERCEPT,
    CURRENT_TO_PEROXIDE_SLOPE,
    CURRENT_UNITS,
    DEFAULT_CURRENT_UNIT,
    DEFAULT_PAYLOAD_TEMPLATE,
    DEFAULT_PROXY_LEVELS,
    MappingConfig,
    PEROXIDE_TO_ECO_INTERCEPT,
    PEROXIDE_TO_ECO_SLOPE,
    SYNTHETIC_ECO_MAX,
    SYNTHETIC_ECO_MIN,
    build_proxy_points,
    convert_current_to_ppm,
    current_for_visible_ppm,
    format_number,
    parse_proxy_levels,
    render_payload_text,
)

APP_BG = "#f4f6fb"
PANEL_BG = "#ffffff"
CURVE_COLOR = "#1f78b4"
PROXY_COLOR = "#ff8c42"
PROXY_CLIPPED_COLOR = "#b45309"
INPUT_COLOR = "#cf2e2e"
GRID_COLOR = "#d9e2ec"
TEXT_MUTED = "#5b6472"
CLIP_COLOR = "#7c8798"


class DemoDayGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Demo-Day GUI | Current to PPM")
        self.root.geometry("1520x980")
        self.root.minsize(1260, 860)
        self.root.configure(bg=APP_BG)

        self.current_value_var = tk.StringVar()
        self.current_unit_var = tk.StringVar(value=DEFAULT_CURRENT_UNIT)
        self.proxy_levels_var = tk.StringVar(
            value=", ".join(format_number(level, 1) for level in DEFAULT_PROXY_LEVELS)
        )
        self.sensitivity_multiplier_var = tk.StringVar(value="1.0")
        self.current_to_peroxide_slope_var = tk.StringVar(
            value=format_number(CURRENT_TO_PEROXIDE_SLOPE, 4)
        )
        self.current_to_peroxide_intercept_var = tk.StringVar(
            value=format_number(CURRENT_TO_PEROXIDE_INTERCEPT, 3)
        )
        self.peroxide_to_ppm_slope_var = tk.StringVar(
            value=format_number(PEROXIDE_TO_ECO_SLOPE, 3)
        )
        self.peroxide_to_ppm_intercept_var = tk.StringVar(
            value=format_number(PEROXIDE_TO_ECO_INTERCEPT, 2)
        )
        self.ppm_min_var = tk.StringVar(value=format_number(SYNTHETIC_ECO_MIN, 1))
        self.ppm_max_var = tk.StringVar(value=format_number(SYNTHETIC_ECO_MAX, 1))

        self.backend_url_var = tk.StringVar(value="http://127.0.0.1:8000/predict")
        self.show_payload_var = tk.BooleanVar(value=True)
        self.show_mapping_controls_var = tk.BooleanVar(value=False)
        self.show_proxy_markers_var = tk.BooleanVar(value=False)

        self.formula_var = tk.StringVar()
        self.summary_var = tk.StringVar(
            value="Enter a PalmSens current and choose its unit to convert it into a PPM value."
        )
        self.backend_status_var = tk.StringVar(
            value="Optional backend panel is ready. Edit the URL and payload template if needed."
        )

        self.mapping_config: MappingConfig | None = MappingConfig()
        self.mapping_error = ""
        self.current_result = None
        self.proxy_points = build_proxy_points(DEFAULT_PROXY_LEVELS)
        self.last_response_text = ""

        self._build_ui()
        self._bind_events()
        self._set_default_template()
        self._toggle_mapping_section()
        self._toggle_proxy_section()
        self._update_all()

    def _build_ui(self) -> None:
        self._build_style()

        shell = ttk.Frame(self.root, style="App.TFrame", padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        top = ttk.Frame(shell, style="Panel.TFrame", padding=18)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="Demo-Day GUI", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            top,
            text=(
                "Standalone desktop viewer for the same current-to-PPM fit used in the "
                "MDIR workflow, with live unit selection and adjustable mapping controls."
            ),
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))
        ttk.Label(
            top,
            textvariable=self.formula_var,
            style="Formula.TLabel",
            justify="left",
        ).grid(row=2, column=0, sticky="w")

        main = ttk.Frame(shell, style="App.TFrame")
        main.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(main, style="Panel.TFrame", padding=18)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Manual current input", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        current_row = ttk.Frame(controls, style="Panel.TFrame")
        current_row.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        current_row.columnconfigure(0, weight=1)

        ttk.Label(current_row, text="PalmSens current value", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=2
        )
        self.current_entry = ttk.Entry(
            current_row,
            textvariable=self.current_value_var,
            font=("Segoe UI", 15),
            width=16,
        )
        self.current_entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.current_unit_combo = ttk.Combobox(
            current_row,
            textvariable=self.current_unit_var,
            values=list(CURRENT_UNITS),
            state="readonly",
            width=8,
        )
        self.current_unit_combo.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(6, 0))
        ttk.Label(
            current_row,
            text="The displayed PPM updates as you type or change units.",
            style="Muted.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(
            controls,
            textvariable=self.summary_var,
            style="Callout.TLabel",
            wraplength=400,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 18))

        ttk.Separator(controls).grid(row=3, column=0, sticky="ew", pady=(0, 18))

        self.mapping_toggle = ttk.Checkbutton(
            controls,
            text="Adaptive mapping controls",
            variable=self.show_mapping_controls_var,
            command=self._toggle_mapping_section,
            style="Dropdown.TCheckbutton",
        )
        self.mapping_toggle.grid(row=4, column=0, sticky="ew")

        self.mapping_section = ttk.Frame(controls, style="Panel.TFrame")
        self.mapping_section.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            self.mapping_section,
            text=(
                "Quick compensation: raise the sensitivity multiplier above 1.0 if the sensor is "
                "reading lower than expected. If you recalculated a new calibration line, edit the "
                "fit constants directly."
            ),
            style="Muted.TLabel",
            wraplength=400,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        mapping_box = ttk.Frame(self.mapping_section, style="Inset.TFrame", padding=14)
        mapping_box.grid(row=1, column=0, sticky="ew")
        for column in range(4):
            mapping_box.columnconfigure(column, weight=1 if column in {1, 3} else 0)

        self._labeled_entry(
            mapping_box,
            row=0,
            column_pair=0,
            label="Sensitivity multiplier",
            variable=self.sensitivity_multiplier_var,
        )
        self._labeled_entry(
            mapping_box,
            row=0,
            column_pair=1,
            label="Current -> peroxide slope",
            variable=self.current_to_peroxide_slope_var,
        )
        self._labeled_entry(
            mapping_box,
            row=1,
            column_pair=0,
            label="Current -> peroxide intercept (uA)",
            variable=self.current_to_peroxide_intercept_var,
        )
        self._labeled_entry(
            mapping_box,
            row=1,
            column_pair=1,
            label="Peroxide -> PPM slope",
            variable=self.peroxide_to_ppm_slope_var,
        )
        self._labeled_entry(
            mapping_box,
            row=2,
            column_pair=0,
            label="Peroxide -> PPM intercept",
            variable=self.peroxide_to_ppm_intercept_var,
        )
        self._labeled_entry(
            mapping_box,
            row=2,
            column_pair=1,
            label="PPM min",
            variable=self.ppm_min_var,
        )
        self._labeled_entry(
            mapping_box,
            row=3,
            column_pair=0,
            label="PPM max",
            variable=self.ppm_max_var,
        )
        ttk.Button(
            mapping_box,
            text="Reset defaults",
            command=self._reset_mapping_defaults,
        ).grid(row=3, column=2, columnspan=2, sticky="w", padx=(14, 0), pady=(14, 0))

        ttk.Separator(controls).grid(row=6, column=0, sticky="ew", pady=(18, 18))

        self.proxy_toggle = ttk.Checkbutton(
            controls,
            text="Adjustable proxy PPM markers",
            variable=self.show_proxy_markers_var,
            command=self._toggle_proxy_section,
            style="Dropdown.TCheckbutton",
        )
        self.proxy_toggle.grid(row=7, column=0, sticky="ew")

        self.proxy_section = ttk.Frame(controls, style="Panel.TFrame")
        self.proxy_section.grid(row=8, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            self.proxy_section,
            text="Comma-separated values to show on the fit curve.",
            style="Muted.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(self.proxy_section, textvariable=self.proxy_levels_var, width=34).grid(
            row=1, column=0, sticky="ew"
        )

        self.proxy_tree = ttk.Treeview(
            self.proxy_section,
            columns=("ppm", "current", "note"),
            show="headings",
            height=7,
        )
        self.proxy_tree.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.proxy_tree.heading("ppm", text="Requested PPM")
        self.proxy_tree.heading("current", text="Current")
        self.proxy_tree.heading("note", text="Note")
        self.proxy_tree.column("ppm", width=110, anchor="center")
        self.proxy_tree.column("current", width=135, anchor="center")
        self.proxy_tree.column("note", width=165, anchor="w")

        plot_panel = ttk.Frame(main, style="Panel.TFrame", padding=18)
        plot_panel.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        plot_panel.columnconfigure(0, weight=1)
        plot_panel.rowconfigure(1, weight=1)

        ttk.Label(plot_panel, text="Fit visualization", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.plot_canvas = tk.Canvas(
            plot_panel,
            background=PANEL_BG,
            highlightthickness=0,
            bd=0,
        )
        self.plot_canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        toggle_row = ttk.Frame(shell, style="App.TFrame", padding=(0, 14, 0, 8))
        toggle_row.grid(row=2, column=0, sticky="ew")
        ttk.Checkbutton(
            toggle_row,
            text="Show backend and JSON payload panel",
            variable=self.show_payload_var,
            command=self._toggle_payload_panel,
        ).pack(anchor="w")

        self.payload_panel = ttk.Frame(shell, style="Panel.TFrame", padding=18)
        self.payload_panel.grid(row=3, column=0, sticky="ew")
        self.payload_panel.columnconfigure(0, weight=1)
        self.payload_panel.columnconfigure(1, weight=1)

        ttk.Label(self.payload_panel, text="Backend sender", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            self.payload_panel,
            text="Use {{PPM}} anywhere in the template to inject the computed numeric PPM value.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))

        url_row = ttk.Frame(self.payload_panel, style="Panel.TFrame")
        url_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        url_row.columnconfigure(1, weight=1)
        ttk.Label(url_row, text="Backend URL", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        ttk.Entry(url_row, textvariable=self.backend_url_var).grid(
            row=0, column=1, sticky="ew"
        )

        ttk.Label(self.payload_panel, text="Payload template", style="Field.TLabel").grid(
            row=3, column=0, sticky="w", pady=(14, 6)
        )
        ttk.Label(
            self.payload_panel,
            text="Preview and last response",
            style="Field.TLabel",
        ).grid(row=3, column=1, sticky="w", pady=(14, 6), padx=(18, 0))

        self.template_text = ScrolledText(
            self.payload_panel,
            wrap="word",
            height=13,
            font=("Consolas", 10),
        )
        self.template_text.grid(row=4, column=0, sticky="nsew")

        self.preview_text = ScrolledText(
            self.payload_panel,
            wrap="word",
            height=13,
            font=("Consolas", 10),
            state="disabled",
        )
        self.preview_text.grid(row=4, column=1, sticky="nsew", padx=(18, 0))

        button_row = ttk.Frame(self.payload_panel, style="Panel.TFrame")
        button_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(button_row, text="Refresh preview", command=self._refresh_preview).pack(
            side="left"
        )
        ttk.Button(button_row, text="Send payload", command=self._send_payload).pack(
            side="left", padx=(10, 0)
        )
        ttk.Label(
            button_row,
            textvariable=self.backend_status_var,
            style="Muted.TLabel",
            wraplength=760,
        ).pack(side="left", padx=(18, 0))

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Inset.TFrame", background="#f9fbfd")
        style.configure("Title.TLabel", background=PANEL_BG, font=("Segoe UI Semibold", 24))
        style.configure("Section.TLabel", background=PANEL_BG, font=("Segoe UI Semibold", 14))
        style.configure("Field.TLabel", background=PANEL_BG, font=("Segoe UI Semibold", 10))
        style.configure(
            "Muted.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Formula.TLabel",
            background=PANEL_BG,
            font=("Consolas", 10),
            foreground="#243447",
        )
        style.configure("Callout.TLabel", background=PANEL_BG, font=("Segoe UI", 11))
        style.configure(
            "MetricLabel.TLabel",
            background="#f9fbfd",
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Dropdown.TCheckbutton",
            background=PANEL_BG,
            font=("Segoe UI Semibold", 14),
        )

    def _labeled_entry(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        column_pair: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        label_column = column_pair * 2
        entry_column = label_column + 1
        ttk.Label(parent, text=label, style="MetricLabel.TLabel").grid(
            row=row,
            column=label_column,
            sticky="w",
            padx=(0, 8 if entry_column == 1 else 14),
            pady=(0 if row == 0 else 14, 0),
        )
        ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row,
            column=entry_column,
            sticky="ew",
            pady=(0 if row == 0 else 14, 0),
        )

    def _bind_events(self) -> None:
        tracked_vars = [
            self.current_value_var,
            self.current_unit_var,
            self.proxy_levels_var,
            self.sensitivity_multiplier_var,
            self.current_to_peroxide_slope_var,
            self.current_to_peroxide_intercept_var,
            self.peroxide_to_ppm_slope_var,
            self.peroxide_to_ppm_intercept_var,
            self.ppm_min_var,
            self.ppm_max_var,
        ]
        for variable in tracked_vars:
            variable.trace_add("write", lambda *_: self._update_all())

        self.plot_canvas.bind("<Configure>", lambda *_: self._draw_plot())
        self.template_text.bind("<<Modified>>", self._on_template_modified)

    def _set_default_template(self) -> None:
        self.template_text.delete("1.0", "end")
        self.template_text.insert("1.0", DEFAULT_PAYLOAD_TEMPLATE)
        self.template_text.edit_modified(False)

    def _on_template_modified(self, _event) -> None:
        if self.template_text.edit_modified():
            self.template_text.edit_modified(False)
            self._refresh_preview()

    def _reset_mapping_defaults(self) -> None:
        self.sensitivity_multiplier_var.set("1.0")
        self.current_to_peroxide_slope_var.set(format_number(CURRENT_TO_PEROXIDE_SLOPE, 4))
        self.current_to_peroxide_intercept_var.set(format_number(CURRENT_TO_PEROXIDE_INTERCEPT, 3))
        self.peroxide_to_ppm_slope_var.set(format_number(PEROXIDE_TO_ECO_SLOPE, 3))
        self.peroxide_to_ppm_intercept_var.set(format_number(PEROXIDE_TO_ECO_INTERCEPT, 2))
        self.ppm_min_var.set(format_number(SYNTHETIC_ECO_MIN, 1))
        self.ppm_max_var.set(format_number(SYNTHETIC_ECO_MAX, 1))

    def _toggle_payload_panel(self) -> None:
        if self.show_payload_var.get():
            self.payload_panel.grid()
            self._refresh_preview()
        else:
            self.payload_panel.grid_remove()

    def _toggle_mapping_section(self) -> None:
        if self.show_mapping_controls_var.get():
            self.mapping_section.grid()
        else:
            self.mapping_section.grid_remove()

    def _toggle_proxy_section(self) -> None:
        if self.show_proxy_markers_var.get():
            self.proxy_section.grid()
        else:
            self.proxy_section.grid_remove()

    def _parse_float(self, text: str, label: str) -> float:
        stripped = text.strip()
        if not stripped:
            raise ValueError(f"{label} is required.")
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc

    def _update_mapping_state(self) -> None:
        try:
            config = MappingConfig(
                sensitivity_multiplier=self._parse_float(
                    self.sensitivity_multiplier_var.get(),
                    "Sensitivity multiplier",
                ),
                current_to_peroxide_slope=self._parse_float(
                    self.current_to_peroxide_slope_var.get(),
                    "Current-to-peroxide slope",
                ),
                current_to_peroxide_intercept=self._parse_float(
                    self.current_to_peroxide_intercept_var.get(),
                    "Current-to-peroxide intercept",
                ),
                peroxide_to_ppm_slope=self._parse_float(
                    self.peroxide_to_ppm_slope_var.get(),
                    "Peroxide-to-PPM slope",
                ),
                peroxide_to_ppm_intercept=self._parse_float(
                    self.peroxide_to_ppm_intercept_var.get(),
                    "Peroxide-to-PPM intercept",
                ),
                ppm_min=self._parse_float(self.ppm_min_var.get(), "PPM min"),
                ppm_max=self._parse_float(self.ppm_max_var.get(), "PPM max"),
            )
            if config.sensitivity_multiplier <= 0:
                raise ValueError("Sensitivity multiplier must be greater than 0.")
            if config.current_to_peroxide_slope == 0:
                raise ValueError("Current-to-peroxide slope cannot be 0.")
            if config.peroxide_to_ppm_slope == 0:
                raise ValueError("Peroxide-to-PPM slope cannot be 0.")
            if config.ppm_max <= config.ppm_min:
                raise ValueError("PPM max must be greater than PPM min.")
        except ValueError as exc:
            self.mapping_config = None
            self.mapping_error = str(exc)
            return

        self.mapping_config = config
        self.mapping_error = ""

    def _parse_current(self) -> tuple[float | None, str | None]:
        text = self.current_value_var.get().strip()
        current_unit = self.current_unit_var.get()
        if not text:
            return None, None
        try:
            current_value = float(text)
        except ValueError:
            return None, f"Enter a numeric current in {current_unit}."
        if current_value < 0:
            return None, "Current cannot be negative."
        return current_value, None

    def _update_all(self) -> None:
        self._update_mapping_state()
        self._update_formula_text()
        self._update_conversion()
        self._update_proxy_table()
        self._draw_plot()
        if self.show_payload_var.get():
            self._refresh_preview()

    def _update_formula_text(self) -> None:
        current_unit = self.current_unit_var.get()
        if self.mapping_config is None:
            self.formula_var.set(f"Fix mapping controls before conversion can run.\n{self.mapping_error}")
            return

        lower_current, _ = current_for_visible_ppm(
            self.mapping_config.ppm_min,
            current_unit=current_unit,
            mapping=self.mapping_config,
        )
        upper_current, _ = current_for_visible_ppm(
            self.mapping_config.ppm_max,
            current_unit=current_unit,
            mapping=self.mapping_config,
        )

        self.formula_var.set(
            "Input current is normalized internally to uA and then mapped with the active settings.\n"
            f"corrected_current_uA = raw_current_uA * {format_number(self.mapping_config.sensitivity_multiplier, 4)}\n"
            f"peroxide_mM = max((corrected_current_uA - {format_number(self.mapping_config.current_to_peroxide_intercept, 6)}) / "
            f"{format_number(self.mapping_config.current_to_peroxide_slope, 6)}, 0)\n"
            f"ppm = clip(({format_number(self.mapping_config.peroxide_to_ppm_slope, 6)} * peroxide_mM) + "
            f"{format_number(self.mapping_config.peroxide_to_ppm_intercept, 6)}, "
            f"{format_number(self.mapping_config.ppm_min, 4)}, {format_number(self.mapping_config.ppm_max, 4)})\n"
            f"Current axis unit: {current_unit} | lower clip near {format_number(lower_current, 6)} {current_unit} | "
            f"upper clip near {format_number(upper_current, 6)} {current_unit}"
        )

    def _clear_results(self) -> None:
        self.current_result = None

    def _update_conversion(self) -> None:
        if self.mapping_config is None:
            self._clear_results()
            self.summary_var.set(f"Fix mapping controls: {self.mapping_error}")
            return

        current_value, error = self._parse_current()
        current_unit = self.current_unit_var.get()
        if error:
            self._clear_results()
            self.summary_var.set(error)
            return

        if current_value is None:
            self._clear_results()
            self.summary_var.set(
                "Enter a PalmSens current and choose its unit. If the sensor is reading low, "
                "raise the sensitivity multiplier above 1.0 or override the fit constants."
            )
            return

        self.current_result = convert_current_to_ppm(
            current_value,
            current_unit=current_unit,
            mapping=self.mapping_config,
        )

        clip_message = (
            "The displayed PPM is clipped by the active mapping range."
            if self.current_result.eco_ppm_clipped
            else "The displayed PPM is inside the active mapping range."
        )
        multiplier_message = (
            f" Sensitivity multiplier = {format_number(self.mapping_config.sensitivity_multiplier, 4)}."
            if self.mapping_config.sensitivity_multiplier != 1.0
            else ""
        )
        self.summary_var.set(
            f"{format_number(self.current_result.input_current_value, 6)} {current_unit} maps to "
            f"{format_number(self.current_result.synthetic_eco_ppm, 4)} ppm.{multiplier_message} {clip_message}"
        )

    def _update_proxy_table(self) -> None:
        self.proxy_tree.heading("current", text=f"Current ({self.current_unit_var.get()})")
        for row_id in self.proxy_tree.get_children():
            self.proxy_tree.delete(row_id)

        if self.mapping_config is None:
            self.proxy_points = []
            self.proxy_tree.insert("", "end", values=("-", "-", self.mapping_error))
            return

        try:
            levels = parse_proxy_levels(self.proxy_levels_var.get())
            self.proxy_points = build_proxy_points(
                levels,
                current_unit=self.current_unit_var.get(),
                mapping=self.mapping_config,
            )
        except ValueError as exc:
            self.proxy_points = []
            self.proxy_tree.insert("", "end", values=("-", "-", str(exc)))
            return

        for point in self.proxy_points:
            note = "Clipped to visible range" if point.clipped_to_fit else ""
            self.proxy_tree.insert(
                "",
                "end",
                values=(
                    format_number(point.requested_ppm, 3),
                    format_number(point.current_value, 6),
                    note,
                ),
            )

    def _current_digits_for_range(self, max_current_value: float) -> int:
        if max_current_value >= 1000:
            return 0
        if max_current_value >= 100:
            return 1
        if max_current_value >= 10:
            return 2
        if max_current_value >= 1:
            return 3
        if max_current_value >= 0.1:
            return 4
        return 6

    def _draw_plot(self) -> None:
        canvas = self.plot_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 320)
        if width < 10 or height < 10:
            return

        left = 78
        right = 36
        top = 30
        bottom = 56
        plot_width = width - left - right
        plot_height = height - top - bottom
        if plot_width <= 0 or plot_height <= 0:
            return

        if self.mapping_config is None:
            canvas.create_text(
                width / 2,
                height / 2,
                text=f"Fix mapping controls before the fit can be drawn.\n{self.mapping_error}",
                fill=INPUT_COLOR,
                width=plot_width - 20,
                justify="center",
            )
            return

        current_unit = self.current_unit_var.get()
        fit_lower_current, _ = current_for_visible_ppm(
            self.mapping_config.ppm_min,
            current_unit=current_unit,
            mapping=self.mapping_config,
        )
        fit_upper_current, _ = current_for_visible_ppm(
            self.mapping_config.ppm_max,
            current_unit=current_unit,
            mapping=self.mapping_config,
        )

        max_proxy_current = max((point.current_value for point in self.proxy_points), default=0.0)
        max_input_current = (
            self.current_result.input_current_value if self.current_result is not None else 0.0
        )
        positive_candidates = [
            value
            for value in [fit_lower_current, fit_upper_current, max_proxy_current, max_input_current]
            if value > 0
        ]
        max_current_value = max(positive_candidates, default=1.0) * 1.15
        max_ppm = self.mapping_config.ppm_max
        min_ppm = self.mapping_config.ppm_min
        current_digits = self._current_digits_for_range(max_current_value)

        def to_x(current_value: float) -> float:
            safe_current_value = max(current_value, 0.0)
            return left + (safe_current_value / max_current_value) * plot_width

        def to_y(ppm_value: float) -> float:
            return top + plot_height - ((ppm_value - min_ppm) / (max_ppm - min_ppm)) * plot_height

        for tick in range(6):
            x = left + (tick / 5) * plot_width
            canvas.create_line(x, top, x, top + plot_height, fill=GRID_COLOR)
            current_label = format_number((tick / 5) * max_current_value, current_digits)
            canvas.create_text(x, top + plot_height + 24, text=current_label, fill=TEXT_MUTED)

        ppm_ticks = [
            min_ppm + ((max_ppm - min_ppm) * tick / 5)
            for tick in range(6)
        ]
        for ppm_value in ppm_ticks:
            y = to_y(ppm_value)
            canvas.create_line(left, y, left + plot_width, y, fill=GRID_COLOR)
            canvas.create_text(left - 34, y, text=format_number(ppm_value, 1), fill=TEXT_MUTED)

        canvas.create_line(left, top, left, top + plot_height, width=2, fill="#1e293b")
        canvas.create_line(
            left,
            top + plot_height,
            left + plot_width,
            top + plot_height,
            width=2,
            fill="#1e293b",
        )
        canvas.create_text(
            left + plot_width / 2,
            height - 20,
            text=f"Current ({current_unit})",
            fill="#1e293b",
        )
        canvas.create_text(26, top + plot_height / 2, text="PPM", angle=90, fill="#1e293b")

        curve_points: list[float] = []
        for index in range(260):
            current_value = (index / 259) * max_current_value
            ppm_value = convert_current_to_ppm(
                current_value,
                current_unit=current_unit,
                mapping=self.mapping_config,
            ).synthetic_eco_ppm
            curve_points.extend((to_x(current_value), to_y(ppm_value)))
        canvas.create_line(curve_points, fill=CURVE_COLOR, width=3, smooth=True)

        lower_x = to_x(fit_lower_current)
        upper_x = to_x(fit_upper_current)
        canvas.create_line(lower_x, top, lower_x, top + plot_height, dash=(4, 4), fill=CLIP_COLOR)
        canvas.create_text(
            lower_x + 6,
            top + 10,
            anchor="nw",
            text=(
                f"{format_number(self.mapping_config.ppm_min, 2)} ppm lower clip: "
                f"{format_number(fit_lower_current, current_digits)} {current_unit}"
            ),
            fill=CLIP_COLOR,
        )
        canvas.create_line(upper_x, top, upper_x, top + plot_height, dash=(4, 4), fill=CLIP_COLOR)
        canvas.create_text(
            upper_x + 6,
            top + 30,
            anchor="nw",
            text=(
                f"{format_number(self.mapping_config.ppm_max, 2)} ppm upper clip: "
                f"{format_number(fit_upper_current, current_digits)} {current_unit}"
            ),
            fill=CLIP_COLOR,
        )

        for point in self.proxy_points:
            x = to_x(point.current_value)
            y = to_y(point.plotted_ppm)
            color = PROXY_CLIPPED_COLOR if point.clipped_to_fit else PROXY_COLOR
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")
            label = f"{format_number(point.requested_ppm, 1)} ppm"
            if point.clipped_to_fit:
                label += " (clipped)"
            canvas.create_text(x + 10, y - 10, text=label, anchor="w", fill=color)

        if self.current_result is not None:
            x = to_x(self.current_result.input_current_value)
            y = to_y(self.current_result.synthetic_eco_ppm)
            canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill=INPUT_COLOR, outline="")
            canvas.create_text(
                x + 12,
                y + 14,
                anchor="w",
                text=(
                    f"Input: {format_number(self.current_result.input_current_value, current_digits)} {current_unit}"
                    f" -> {format_number(self.current_result.synthetic_eco_ppm, 3)} ppm"
                ),
                fill=INPUT_COLOR,
            )

        legend_x = left + 16
        legend_y = top + 16
        canvas.create_text(legend_x, legend_y, anchor="nw", text="Blue: fit", fill=CURVE_COLOR)
        canvas.create_text(
            legend_x,
            legend_y + 20,
            anchor="nw",
            text="Orange: proxy markers",
            fill=PROXY_COLOR,
        )
        canvas.create_text(
            legend_x,
            legend_y + 40,
            anchor="nw",
            text="Red: entered current",
            fill=INPUT_COLOR,
        )

    def _get_template_text(self) -> str:
        return self.template_text.get("1.0", "end-1c")

    def _set_preview_text(self, text: str) -> None:
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    def _refresh_preview(self) -> None:
        if not self.show_payload_var.get():
            return
        if self.current_result is None:
            self._set_preview_text("Enter a numeric current to build the payload preview.")
            return

        template = self._get_template_text().strip()
        if not template:
            self._set_preview_text("Payload template is empty.")
            return

        try:
            rendered = render_payload_text(template, self.current_result.synthetic_eco_ppm)
        except ValueError as exc:
            self._set_preview_text(str(exc))
            return

        pretty_rendered = json.dumps(json.loads(rendered), indent=2)
        sections = [f"Request preview\n{pretty_rendered}"]
        if self.last_response_text:
            sections.append(self.last_response_text)
        self._set_preview_text("\n\n".join(sections))

    def _send_payload(self) -> None:
        if self.current_result is None:
            messagebox.showerror("Cannot send payload", "Enter a valid current first.")
            return

        backend_url = self.backend_url_var.get().strip()
        if not backend_url:
            messagebox.showerror("Cannot send payload", "Enter a backend URL first.")
            return

        try:
            rendered = render_payload_text(
                self._get_template_text(),
                self.current_result.synthetic_eco_ppm,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid payload template", str(exc))
            return

        self.backend_status_var.set(
            f"Sending {format_number(self.current_result.synthetic_eco_ppm, 4)} ppm to {backend_url}..."
        )

        thread = threading.Thread(
            target=self._send_payload_worker,
            args=(backend_url, rendered),
            daemon=True,
        )
        thread.start()

    def _send_payload_worker(self, backend_url: str, rendered_payload: str) -> None:
        request = urllib.request.Request(
            backend_url,
            data=rendered_payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                status = response.status
                raw_body = response.read().decode("utf-8", errors="replace")
                success = True
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw_body = exc.read().decode("utf-8", errors="replace")
            success = False
        except Exception as exc:  # noqa: BLE001
            self.root.after(
                0,
                lambda: self._finish_send(
                    False,
                    "Request failed before a response was returned.",
                    str(exc),
                ),
            )
            return

        self.root.after(
            0,
            lambda: self._finish_send(success, f"HTTP {status}", raw_body),
        )

    def _finish_send(self, success: bool, headline: str, raw_body: str) -> None:
        response_body = raw_body
        try:
            response_body = json.dumps(json.loads(raw_body), indent=2)
        except (TypeError, ValueError):
            response_body = raw_body

        self.last_response_text = f"Last response\n{headline}\n{response_body}"
        self._refresh_preview()

        if success:
            self.backend_status_var.set(f"Payload sent successfully. {headline}.")
        else:
            self.backend_status_var.set(f"Payload send failed. {headline}.")


def main() -> None:
    root = tk.Tk()
    app = DemoDayGuiApp(root)
    app.current_entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    main()
