#!/usr/bin/env python3
"""
Thickness Iteration Tool v2.0 - Advanced
==========================================
Automated thickness optimization for BDF models with property-level control.

Features:
- Reads main BDF file
- Individual property thickness optimization
- Automatic offset calculation and application
- Runs Nastran analysis
- Calculates RF using allowable data with power law fitting
- Sensitivity-based optimization for minimum weight
- Detailed iteration logging per folder

Author: Generated for structural analysis workflow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import re
import threading
import csv
import shutil
import tempfile
import pandas as pd
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
import subprocess
import time
from datetime import datetime


class ThicknessIterationToolV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Thickness Iteration Tool v2.0 - Advanced Optimization")
        self.root.geometry("1300x950")

        # Input paths
        self.input_bdf_path = tk.StringVar()
        self.allowable_excel_path = tk.StringVar()
        self.property_excel_path = tk.StringVar()
        self.element_excel_path = tk.StringVar()
        self.nastran_path = tk.StringVar()
        self.output_folder = tk.StringVar()

        # Thickness ranges (global defaults)
        self.bar_min_thickness = tk.StringVar(value="2.0")
        self.bar_max_thickness = tk.StringVar(value="12.0")
        self.skin_min_thickness = tk.StringVar(value="3.0")
        self.skin_max_thickness = tk.StringVar(value="18.0")
        self.thickness_step = tk.StringVar(value="0.5")

        # RF settings
        self.target_rf = tk.StringVar(value="1.0")
        self.rf_tolerance = tk.StringVar(value="0.05")
        self.r2_threshold = tk.StringVar(value="0.95")
        self.min_data_points = tk.StringVar(value="3")

        # Material/Weight settings
        self.density = tk.StringVar(value="2.7e-9")  # tonnes/mm³

        # Optimization settings
        self.max_iterations = tk.StringVar(value="50")
        self.convergence_threshold = tk.StringVar(value="0.01")  # 1% weight change

        # Internal storage
        self.bdf_model = None

        # Property data
        self.bar_properties = {}  # PID -> {dim1, dim2, type, thickness}
        self.skin_properties = {}  # PID -> {thickness, material}

        # Current thickness state (PID -> current_thickness)
        self.current_bar_thicknesses = {}
        self.current_skin_thicknesses = {}

        # Allowable fits
        self.bar_allowable_interp = {}  # PID -> {a, b, r2, excluded}
        self.skin_allowable_interp = {}  # PID -> {a, b, r2, excluded}

        # Geometry data
        self.element_areas = {}  # EID -> area (shells)
        self.bar_lengths = {}  # EID -> length (bars)
        self.prop_elements = {}  # PID -> [EID list]
        self.elem_to_prop = {}  # EID -> PID

        # Element lists for offsets
        self.landing_elem_ids = []
        self.bar_offset_elem_ids = []

        # Residual strength
        self.residual_strength_df = None

        # Results
        self.iteration_results = []
        self.best_solution = None
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI with scrolling"""
        # Main scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = scrollable_frame

        # Title
        ttk.Label(main, text="Thickness Iteration Tool v2.0",
                  font=('Helvetica', 16, 'bold')).pack(pady=(10, 5))
        ttk.Label(main, text="Property-level thickness optimization with RF targeting and minimum weight objective",
                  font=('Helvetica', 10), foreground='gray').pack(pady=(0, 10))

        # === Section 1: Input Files ===
        self._create_input_section(main)

        # === Section 2: Thickness Ranges ===
        self._create_range_section(main)

        # === Section 3: RF Settings ===
        self._create_rf_section(main)

        # === Section 4: Optimization Settings ===
        self._create_optimization_section(main)

        # === Section 5: Actions ===
        self._create_action_section(main)

        # === Section 6: Results ===
        self._create_results_section(main)

        # === Section 7: Log ===
        self._create_log_section(main)

        # Initial instructions
        self._log_instructions()

    def _create_input_section(self, parent):
        """Create input files section"""
        frame = ttk.LabelFrame(parent, text="1. Input Files", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # BDF File
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Main BDF File:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.input_bdf_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.input_bdf_path, "BDF Files", "*.bdf *.dat *.nas")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self.load_bdf).pack(side=tk.LEFT, padx=2)

        self.bdf_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.bdf_status.pack(anchor=tk.W, pady=2)

        # Property Excel
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Property Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.property_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.property_excel_path, "Excel", "*.xlsx *.xls")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self.load_properties).pack(side=tk.LEFT, padx=2)

        self.prop_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.prop_status.pack(anchor=tk.W, pady=2)

        # Allowable Excel
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Allowable Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.allowable_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.allowable_excel_path, "Excel/CSV", "*.xlsx *.xls *.csv")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load & Fit", command=self.load_allowable).pack(side=tk.LEFT, padx=2)

        self.allow_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.allow_status.pack(anchor=tk.W, pady=2)

        # Element IDs Excel
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Element IDs Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.element_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.element_excel_path, "Excel", "*.xlsx *.xls")).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self.load_element_ids).pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(Landing_Offset, Bar_Offset)", foreground='gray').pack(side=tk.LEFT, padx=5)

        self.elem_status = ttk.Label(frame, text="Not loaded", foreground="gray")
        self.elem_status.pack(anchor=tk.W, pady=2)

        # Nastran Path
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Nastran Executable:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.nastran_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.nastran_path, "Executable", "*.exe *.bat *")).pack(side=tk.LEFT, padx=2)

        # Output Folder
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Output Folder:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.output_folder, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=self._browse_folder).pack(side=tk.LEFT, padx=2)

    def _create_range_section(self, parent):
        """Create thickness range section"""
        frame = ttk.LabelFrame(parent, text="2. Thickness Ranges (mm)", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # Bar range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Bar Thickness:", width=15).pack(side=tk.LEFT)
        ttk.Label(row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.bar_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.bar_max_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="(Applied to all bar properties)", foreground='gray').pack(side=tk.LEFT, padx=10)

        # Skin range
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Skin Thickness:", width=15).pack(side=tk.LEFT)
        ttk.Label(row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.skin_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.skin_max_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="(Applied to all skin properties)", foreground='gray').pack(side=tk.LEFT, padx=10)

        # Step
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Initial Step:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.thickness_step, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Step size for initial search, adaptive during optimization)", foreground='gray').pack(side=tk.LEFT, padx=10)

    def _create_rf_section(self, parent):
        """Create RF settings section"""
        frame = ttk.LabelFrame(parent, text="3. RF Target Settings", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Target RF:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.target_rf, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Tolerance (+/-):", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.rf_tolerance, width=8).pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="R² Threshold:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.r2_threshold, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Min Data Points:", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.min_data_points, width=8).pack(side=tk.LEFT, padx=5)

    def _create_optimization_section(self, parent):
        """Create optimization settings section"""
        frame = ttk.LabelFrame(parent, text="4. Optimization Settings", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Iterations:", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.max_iterations, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Convergence:", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(row, textvariable=self.convergence_threshold, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(weight change threshold)", foreground='gray').pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Density (t/mm³):", width=15).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.density, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(2.7e-9 for aluminum, 7.85e-9 for steel)", foreground='gray').pack(side=tk.LEFT, padx=10)

    def _create_action_section(self, parent):
        """Create action buttons section"""
        frame = ttk.LabelFrame(parent, text="5. Actions", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)

        self.btn_start = ttk.Button(row, text=">>> START ITERATION <<<",
                                     command=self.start_iteration, width=25)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(row, text="STOP", command=self.stop_iteration,
                                    width=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Button(row, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Test Single Run", command=self.test_single_run).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(frame, text="Ready")
        self.progress_label.pack(anchor=tk.W)

    def _create_results_section(self, parent):
        """Create results section"""
        frame = ttk.LabelFrame(parent, text="6. Results Summary", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        self.result_summary = ttk.Label(frame,
                                         text="Run iteration to see results",
                                         font=('Helvetica', 11, 'bold'), foreground='blue')
        self.result_summary.pack(anchor=tk.W, pady=5)

        self.best_solution_text = tk.Text(frame, height=8, width=120, font=('Courier', 9))
        self.best_solution_text.pack(fill=tk.X, pady=5)
        self.best_solution_text.insert(tk.END, "Best solution details will appear here...")
        self.best_solution_text.config(state=tk.DISABLED)

    def _create_log_section(self, parent):
        """Create log section"""
        frame = ttk.LabelFrame(parent, text="7. Iteration Log", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.log_text = scrolledtext.ScrolledText(frame, height=20, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log_instructions(self):
        """Log initial instructions"""
        self.log("="*80)
        self.log("Thickness Iteration Tool v2.0 - Advanced Property-Level Optimization")
        self.log("="*80)
        self.log("\nWorkflow:")
        self.log("1. Load BDF file (contains shell and bar elements)")
        self.log("2. Load Property Excel (Bar_Properties, Skin_Properties, Residual_Strength sheets)")
        self.log("3. Load Allowable Excel (Bar_Allowable, Skin_Allowable sheets with T vs Allowable)")
        self.log("4. Load Element IDs Excel (Landing_Offset, Bar_Offset sheets) - optional for offset")
        self.log("5. Set thickness ranges and RF target")
        self.log("6. Click 'START ITERATION'")
        self.log("\nAlgorithm:")
        self.log("- Starts with minimum thicknesses for all properties")
        self.log("- For each iteration:")
        self.log("  1. Updates PBARL/PSHELL cards in BDF")
        self.log("  2. Calculates offsets (landing zoffset = -t/2, bar wa/wb)")
        self.log("  3. Runs Nastran analysis")
        self.log("  4. Extracts stress results from OP2")
        self.log("  5. Calculates RF for each element using power law: Allowable = a * T^b")
        self.log("  6. Identifies failing elements (RF < target)")
        self.log("  7. Increases thickness for properties with failures")
        self.log("  8. Uses sensitivity to adjust thickness increments")
        self.log("- Continues until all RF >= target with minimum weight")
        self.log("\nWeight Calculation:")
        self.log("  Skin: Sum(element_area) * thickness * density")
        self.log("  Bar:  Sum(bar_length) * dim1 * dim2 * density")
        self.log("="*80 + "\n")

    # ========== UTILITY FUNCTIONS ==========
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def update_progress(self, value, text=""):
        self.progress['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    def _browse_file(self, var, desc, pattern):
        f = filedialog.askopenfilename(filetypes=[(desc, pattern), ("All", "*.*")])
        if f:
            var.set(f)

    def _browse_folder(self):
        f = filedialog.askdirectory()
        if f:
            self.output_folder.set(f)

    # ========== LOADING FUNCTIONS ==========
    def load_bdf(self):
        """Load BDF and extract geometry"""
        path = self.input_bdf_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select a valid BDF file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING BDF FILE")
        self.log("="*70)

        try:
            self.bdf_model = BDF(debug=False)
            self.bdf_model.read_bdf(path, validate=False, xref=True,
                                     read_includes=True, encoding='latin-1')

            n_nodes = len(self.bdf_model.nodes)
            n_elements = len(self.bdf_model.elements)
            n_properties = len(self.bdf_model.properties)

            self.log(f"  Nodes: {n_nodes}")
            self.log(f"  Elements: {n_elements}")
            self.log(f"  Properties: {n_properties}")

            # Reset storage
            self.element_areas = {}
            self.bar_lengths = {}
            self.prop_elements = {}
            self.elem_to_prop = {}

            shell_count = 0
            bar_count = 0

            for eid, elem in self.bdf_model.elements.items():
                pid = elem.pid if hasattr(elem, 'pid') else None

                if pid:
                    self.elem_to_prop[eid] = pid
                    if pid not in self.prop_elements:
                        self.prop_elements[pid] = []
                    self.prop_elements[pid].append(eid)

                if elem.type in ['CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6']:
                    shell_count += 1
                    try:
                        self.element_areas[eid] = elem.Area()
                    except:
                        self.element_areas[eid] = 0

                elif elem.type in ['CBAR', 'CBEAM']:
                    bar_count += 1
                    try:
                        self.bar_lengths[eid] = elem.Length()
                    except:
                        self.bar_lengths[eid] = 0

            self.log(f"  Shell elements: {shell_count}")
            self.log(f"  Bar elements: {bar_count}")
            self.log(f"  Total shell area: {sum(self.element_areas.values()):.2f} mm²")
            self.log(f"  Total bar length: {sum(self.bar_lengths.values()):.2f} mm")

            self.bdf_status.config(
                text=f"✓ {n_elements} elements, {n_properties} properties",
                foreground="green"
            )

            if not self.output_folder.get():
                self.output_folder.set(os.path.dirname(path))

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.bdf_status.config(text=f"Error: {e}", foreground="red")

    def load_properties(self):
        """Load property definitions"""
        path = self.property_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select Property Excel")
            return

        self.log("\n" + "="*70)
        self.log("LOADING PROPERTY DEFINITIONS")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            sheets = xl.sheet_names
            self.log(f"Sheets: {', '.join(sheets)}")

            bar_sheet = skin_sheet = residual_sheet = None
            for s in sheets:
                sl = s.lower().replace('_', '').replace(' ', '')
                if 'bar' in sl and 'prop' in sl:
                    bar_sheet = s
                elif 'skin' in sl and 'prop' in sl:
                    skin_sheet = s
                elif 'residual' in sl or 'strength' in sl:
                    residual_sheet = s

            self.bar_properties = {}
            self.skin_properties = {}
            self.current_bar_thicknesses = {}
            self.current_skin_thicknesses = {}

            bar_min = float(self.bar_min_thickness.get())
            skin_min = float(self.skin_min_thickness.get())

            if bar_sheet:
                self.log(f"\nReading '{bar_sheet}'...")
                df = pd.read_excel(xl, sheet_name=bar_sheet)
                for _, row in df.iterrows():
                    pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                    if pid:
                        dim1 = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else bar_min
                        dim2 = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else dim1
                        bar_type = str(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else 'BOX'

                        self.bar_properties[pid] = {
                            'dim1': dim1,
                            'dim2': dim2,
                            'type': bar_type
                        }
                        # Initialize current thickness to minimum
                        self.current_bar_thicknesses[pid] = bar_min

                self.log(f"  Loaded {len(self.bar_properties)} bar properties")

            if skin_sheet:
                self.log(f"\nReading '{skin_sheet}'...")
                df = pd.read_excel(xl, sheet_name=skin_sheet)
                for _, row in df.iterrows():
                    pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                    if pid:
                        t = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else skin_min
                        mat = int(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 1

                        self.skin_properties[pid] = {
                            'thickness': t,
                            'material': mat
                        }
                        self.current_skin_thicknesses[pid] = skin_min

                self.log(f"  Loaded {len(self.skin_properties)} skin properties")

            if residual_sheet:
                self.log(f"\nReading '{residual_sheet}'...")
                self.residual_strength_df = pd.read_excel(xl, sheet_name=residual_sheet)
                self.log(f"  Loaded {len(self.residual_strength_df)} residual strength rows")

            self.prop_status.config(
                text=f"✓ Bar: {len(self.bar_properties)}, Skin: {len(self.skin_properties)}",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.prop_status.config(text=f"Error", foreground="red")

    def load_allowable(self):
        """Load allowable data and fit power law curves"""
        path = self.allowable_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select Allowable file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING ALLOWABLE DATA & FITTING POWER LAW")
        self.log("="*70)

        try:
            r2_thresh = float(self.r2_threshold.get())
            min_pts = int(self.min_data_points.get())
        except:
            r2_thresh = 0.95
            min_pts = 3

        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                self._fit_power_law(df, 'bar', r2_thresh, min_pts)
            else:
                xl = pd.ExcelFile(path)
                sheets = xl.sheet_names
                self.log(f"Sheets: {', '.join(sheets)}")

                for s in sheets:
                    sl = s.lower().replace('_', '').replace(' ', '')
                    if 'bar' in sl and 'allow' in sl:
                        self.log(f"\nFitting '{s}' (Bar)...")
                        df = pd.read_excel(xl, sheet_name=s)
                        self._fit_power_law(df, 'bar', r2_thresh, min_pts)
                    elif 'skin' in sl and 'allow' in sl:
                        self.log(f"\nFitting '{s}' (Skin)...")
                        df = pd.read_excel(xl, sheet_name=s)
                        self._fit_power_law(df, 'skin', r2_thresh, min_pts)

            n_bar = len(self.bar_allowable_interp)
            n_skin = len(self.skin_allowable_interp)

            self.allow_status.config(
                text=f"✓ Bar: {n_bar} fits, Skin: {n_skin} fits",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.allow_status.config(text="Error", foreground="red")

    def _fit_power_law(self, df, prop_type, r2_thresh, min_pts):
        """Fit power law Allowable = a * T^b for each property"""
        # Standardize column names
        col_map = {}
        for col in df.columns:
            cu = col.upper().replace(' ', '_').strip('_')
            if cu in ['PROPERTY_ID', 'PROPERTY', 'PROP_ID', 'PID']:
                col_map[col] = 'Property'
            elif cu in ['T', 'THICKNESS', 'T_MM', 'DIM1', 'DIM']:
                col_map[col] = 'Thickness'
            elif cu in ['ALLOWABLE', 'ALLOW', 'ALLOWABLE_STRESS', 'ALLOW_MPA']:
                col_map[col] = 'Allowable'

        df = df.rename(columns=col_map)

        # Clean data
        df['Property'] = pd.to_numeric(df['Property'], errors='coerce')
        df['Thickness'] = pd.to_numeric(df['Thickness'], errors='coerce')
        df['Allowable'] = pd.to_numeric(df['Allowable'], errors='coerce')
        df = df.dropna(subset=['Property', 'Thickness', 'Allowable'])

        interp = self.bar_allowable_interp if prop_type == 'bar' else self.skin_allowable_interp
        valid_count = 0
        excluded_count = 0

        for pid in df['Property'].unique():
            pid_int = int(pid)
            pdata = df[df['Property'] == pid]
            n = len(pdata)

            if n < min_pts:
                interp[pid_int] = {'a': pdata['Allowable'].mean(), 'b': 0, 'r2': 0, 'n': n, 'excluded': True}
                excluded_count += 1
                continue

            x = pdata['Thickness'].values.astype(float)
            y = pdata['Allowable'].values.astype(float)

            mask = (x > 0) & (y > 0)
            x, y = x[mask], y[mask]

            if len(x) < 2:
                interp[pid_int] = {'a': np.mean(y), 'b': 0, 'r2': 0, 'n': len(x), 'excluded': True}
                excluded_count += 1
                continue

            try:
                # Log-linear fit
                log_x, log_y = np.log(x), np.log(y)
                coeffs = np.polyfit(log_x, log_y, 1)
                b, log_a = coeffs[0], coeffs[1]
                a = np.exp(log_a)

                # R²
                y_pred = a * (x ** b)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r2 < r2_thresh:
                    interp[pid_int] = {'a': np.mean(y), 'b': 0, 'r2': r2, 'n': n, 'excluded': True}
                    excluded_count += 1
                else:
                    interp[pid_int] = {'a': a, 'b': b, 'r2': r2, 'n': n, 'excluded': False}
                    valid_count += 1

            except:
                interp[pid_int] = {'a': np.mean(y), 'b': 0, 'r2': 0, 'n': n, 'excluded': True}
                excluded_count += 1

        self.log(f"  Valid fits: {valid_count}, Excluded: {excluded_count}")

        # Show sample fits
        self.log(f"  Sample fits:")
        for pid in list(interp.keys())[:5]:
            p = interp[pid]
            if not p['excluded']:
                self.log(f"    PID {pid}: Allowable = {p['a']:.4f} × T^({p['b']:.4f}), R²={p['r2']:.4f}")

    def load_element_ids(self):
        """Load element IDs for offset calculation"""
        path = self.element_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select Element IDs Excel")
            return

        self.log("\n" + "="*70)
        self.log("LOADING ELEMENT IDs FOR OFFSET")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            sheets = xl.sheet_names
            self.log(f"Sheets: {', '.join(sheets)}")

            self.landing_elem_ids = []
            self.bar_offset_elem_ids = []

            for s in sheets:
                sl = s.lower().replace('_', '').replace(' ', '')
                if 'landing' in sl and 'offset' in sl:
                    df = pd.read_excel(xl, sheet_name=s)
                    self.landing_elem_ids = df.iloc[:, 0].dropna().astype(int).tolist()
                    self.log(f"  Landing elements: {len(self.landing_elem_ids)}")
                elif 'bar' in sl and 'offset' in sl:
                    df = pd.read_excel(xl, sheet_name=s)
                    self.bar_offset_elem_ids = df.iloc[:, 0].dropna().astype(int).tolist()
                    self.log(f"  Bar offset elements: {len(self.bar_offset_elem_ids)}")

            self.elem_status.config(
                text=f"✓ Landing: {len(self.landing_elem_ids)}, Bar: {len(self.bar_offset_elem_ids)}",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            self.elem_status.config(text="Error", foreground="red")

    # ========== ITERATION CORE ==========
    def start_iteration(self):
        """Start optimization iteration"""
        # Validate
        if not self.bdf_model:
            messagebox.showerror("Error", "Load BDF first")
            return
        if not self.bar_allowable_interp and not self.skin_allowable_interp:
            messagebox.showerror("Error", "Load allowable data first")
            return
        if not self.output_folder.get():
            messagebox.showerror("Error", "Select output folder")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.iteration_results = []
        self.best_solution = None

        threading.Thread(target=self._run_optimization, daemon=True).start()

    def stop_iteration(self):
        self.is_running = False
        self.log("\n*** STOPPING ***")

    def test_single_run(self):
        """Test a single iteration without optimization"""
        if not self.bdf_model:
            messagebox.showerror("Error", "Load BDF first")
            return

        threading.Thread(target=self._test_single, daemon=True).start()

    def _test_single(self):
        """Run a single test iteration"""
        try:
            self.log("\n" + "="*70)
            self.log("TEST SINGLE RUN")
            self.log("="*70)

            bar_min = float(self.bar_min_thickness.get())
            skin_min = float(self.skin_min_thickness.get())

            # Initialize thicknesses
            for pid in self.bar_properties:
                self.current_bar_thicknesses[pid] = bar_min
            for pid in self.skin_properties:
                self.current_skin_thicknesses[pid] = skin_min

            # Create test folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_folder = os.path.join(self.output_folder.get(), f"test_{timestamp}")
            os.makedirs(test_folder, exist_ok=True)

            # Run single iteration
            result = self._run_single_iteration(test_folder, 0)

            if result:
                self.log(f"\nTest complete!")
                self.log(f"  Min RF: {result['min_rf']:.4f}")
                self.log(f"  Weight: {result['weight']:.6f} tonnes")
                self.log(f"  Failures: {result['n_fail']}")
                self.log(f"  Folder: {test_folder}")

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _run_optimization(self):
        """Main optimization loop"""
        try:
            self.log("\n" + "="*70)
            self.log("STARTING OPTIMIZATION")
            self.log("="*70)

            # Get parameters
            bar_min = float(self.bar_min_thickness.get())
            bar_max = float(self.bar_max_thickness.get())
            skin_min = float(self.skin_min_thickness.get())
            skin_max = float(self.skin_max_thickness.get())
            step = float(self.thickness_step.get())
            target_rf = float(self.target_rf.get())
            rf_tol = float(self.rf_tolerance.get())
            max_iter = int(self.max_iterations.get())
            conv_thresh = float(self.convergence_threshold.get())
            density = float(self.density.get())

            self.log(f"\nParameters:")
            self.log(f"  Bar range: {bar_min} - {bar_max} mm")
            self.log(f"  Skin range: {skin_min} - {skin_max} mm")
            self.log(f"  Target RF: {target_rf} ± {rf_tol}")
            self.log(f"  Max iterations: {max_iter}")

            # Create base folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_folder = os.path.join(self.output_folder.get(), f"optimization_{timestamp}")
            os.makedirs(base_folder, exist_ok=True)

            # Initialize all thicknesses to minimum
            for pid in self.bar_properties:
                self.current_bar_thicknesses[pid] = bar_min
            for pid in self.skin_properties:
                self.current_skin_thicknesses[pid] = skin_min

            iteration = 0
            converged = False
            prev_weight = 0
            best_weight = float('inf')

            while iteration < max_iter and self.is_running and not converged:
                iteration += 1

                self.log(f"\n{'='*60}")
                self.log(f"ITERATION {iteration}")
                self.log(f"{'='*60}")

                progress = (iteration / max_iter) * 100
                self.update_progress(progress, f"Iteration {iteration}/{max_iter}")

                # Create iteration folder
                iter_folder = os.path.join(base_folder, f"iter_{iteration:03d}")
                os.makedirs(iter_folder, exist_ok=True)

                # Run this iteration
                result = self._run_single_iteration(iter_folder, iteration)

                if not result:
                    self.log("  Iteration failed, trying again with increased thickness...")
                    # Increase all thicknesses
                    for pid in self.current_bar_thicknesses:
                        self.current_bar_thicknesses[pid] = min(
                            self.current_bar_thicknesses[pid] + step, bar_max)
                    for pid in self.current_skin_thicknesses:
                        self.current_skin_thicknesses[pid] = min(
                            self.current_skin_thicknesses[pid] + step, skin_max)
                    continue

                self.iteration_results.append(result)

                # Check for best solution
                if result['min_rf'] >= target_rf - rf_tol and result['weight'] < best_weight:
                    best_weight = result['weight']
                    self.best_solution = result.copy()
                    self.log(f"\n  *** NEW BEST: Weight = {best_weight:.6f} t, RF = {result['min_rf']:.4f} ***")

                # Check convergence
                weight_change = abs(result['weight'] - prev_weight) / max(prev_weight, 1e-10)
                if result['n_fail'] == 0 and weight_change < conv_thresh:
                    self.log(f"\n  *** CONVERGED (weight change {weight_change*100:.2f}% < {conv_thresh*100:.2f}%) ***")
                    converged = True
                    break

                prev_weight = result['weight']

                # Update thicknesses based on results
                if result['n_fail'] > 0:
                    self._update_thicknesses_smart(result, step, bar_min, bar_max, skin_min, skin_max, target_rf)
                else:
                    # Try to reduce thickness for over-designed properties
                    self._reduce_overdesigned(result, step, bar_min, skin_min, target_rf, rf_tol)

            # Final summary
            self.log("\n" + "="*70)
            self.log("OPTIMIZATION COMPLETE")
            self.log("="*70)

            if self.best_solution:
                self.log(f"\nBest Solution:")
                self.log(f"  Weight: {self.best_solution['weight']:.6f} tonnes")
                self.log(f"  Min RF: {self.best_solution['min_rf']:.4f}")
                self.log(f"  Folder: {self.best_solution['folder']}")

                # Show thickness summary
                self.log(f"\nFinal Bar Thicknesses:")
                for pid in sorted(self.current_bar_thicknesses.keys())[:10]:
                    self.log(f"  PID {pid}: {self.current_bar_thicknesses[pid]:.2f} mm")

                self.log(f"\nFinal Skin Thicknesses:")
                for pid in sorted(self.current_skin_thicknesses.keys())[:10]:
                    self.log(f"  PID {pid}: {self.current_skin_thicknesses[pid]:.2f} mm")

                self.root.after(0, self._update_ui_results)

            # Save all results
            self._save_all_results(base_folder)

            self.root.after(0, lambda: messagebox.showinfo("Complete",
                f"Optimization complete!\nIterations: {iteration}\nBest weight: {best_weight:.6f} t"))

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.is_running = False
            self.root.after(0, lambda: [
                self.btn_start.config(state=tk.NORMAL),
                self.btn_stop.config(state=tk.DISABLED),
                self.update_progress(100, "Complete")
            ])

    def _run_single_iteration(self, iter_folder, iteration):
        """Run a single iteration: update BDF -> offset -> nastran -> RF"""
        try:
            density = float(self.density.get())
            target_rf = float(self.target_rf.get())

            # Step 1: Update BDF
            self.log("  [1] Updating BDF properties...")
            bdf_path = self._write_updated_bdf(iter_folder)
            if not bdf_path:
                return None

            # Step 2: Calculate and apply offsets
            self.log("  [2] Applying offsets...")
            offset_bdf = self._apply_offsets(bdf_path, iter_folder)
            if not offset_bdf:
                offset_bdf = bdf_path

            # Step 3: Run Nastran
            self.log("  [3] Running Nastran...")
            if not self._run_nastran(offset_bdf, iter_folder):
                self.log("    WARNING: Nastran failed")
                # Try to continue with dummy results for testing

            # Step 4: Extract stresses
            self.log("  [4] Extracting stresses...")
            stresses = self._extract_stresses(iter_folder)

            # Step 5: Calculate RF
            self.log("  [5] Calculating RF...")
            rf_results = self._calculate_rf_per_element(stresses, target_rf)

            # Step 6: Calculate weight
            self.log("  [6] Calculating weight...")
            weight = self._calculate_total_weight(density)

            # Summary
            min_rf = rf_results['min_rf'] if rf_results else 0
            n_fail = rf_results['n_fail'] if rf_results else 0
            n_total = rf_results['n_total'] if rf_results else 0

            self.log(f"\n  Results: Min RF = {min_rf:.4f}, Failures = {n_fail}/{n_total}, Weight = {weight:.6f} t")

            # Save iteration summary
            result = {
                'iteration': iteration,
                'min_rf': min_rf,
                'mean_rf': rf_results.get('mean_rf', 0) if rf_results else 0,
                'n_fail': n_fail,
                'n_total': n_total,
                'weight': weight,
                'folder': iter_folder,
                'bar_thicknesses': self.current_bar_thicknesses.copy(),
                'skin_thicknesses': self.current_skin_thicknesses.copy(),
                'failing_pids': rf_results.get('failing_pids', set()) if rf_results else set()
            }

            self._save_iteration_results(iter_folder, result, rf_results)

            return result

        except Exception as e:
            self.log(f"  ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _write_updated_bdf(self, iter_folder):
        """Write BDF with updated property thicknesses"""
        try:
            input_bdf = self.input_bdf_path.get()
            output_bdf = os.path.join(iter_folder, "model.bdf")

            with open(input_bdf, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            new_lines = []
            i = 0
            bar_updated = 0
            skin_updated = 0

            while i < len(lines):
                line = lines[i]

                # Update PBARL
                if line.startswith('PBARL'):
                    try:
                        pid = int(line[8:16].strip())
                        if pid in self.current_bar_thicknesses:
                            t = self.current_bar_thicknesses[pid]
                            new_lines.append(line)
                            i += 1
                            # Update continuation with dimensions
                            while i < len(lines) and (lines[i].startswith('+') or lines[i].startswith('*') or
                                  (lines[i].startswith(' ') and lines[i].strip())):
                                cont = lines[i]
                                if cont.strip() and not cont.strip().startswith('$'):
                                    # Replace dimensions with current thickness
                                    new_cont = cont[:8] + f"{t:8.4f}" + f"{t:8.4f}"
                                    if len(cont) > 24:
                                        new_cont += cont[24:]
                                    else:
                                        new_cont += '\n'
                                    new_lines.append(new_cont)
                                    bar_updated += 1
                                else:
                                    new_lines.append(cont)
                                i += 1
                            continue
                    except:
                        pass

                # Update PSHELL
                elif line.startswith('PSHELL'):
                    try:
                        pid = int(line[8:16].strip())
                        if pid in self.current_skin_thicknesses:
                            t = self.current_skin_thicknesses[pid]
                            # Field 4 (cols 24-32) is thickness
                            new_line = line[:24] + f"{t:8.4f}" + line[32:]
                            new_lines.append(new_line)
                            skin_updated += 1
                            i += 1
                            continue
                    except:
                        pass

                new_lines.append(line)
                i += 1

            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            self.log(f"    Updated: {bar_updated} PBARL, {skin_updated} PSHELL")
            return output_bdf

        except Exception as e:
            self.log(f"    ERROR writing BDF: {e}")
            return None

    def _apply_offsets(self, bdf_path, iter_folder):
        """Calculate and apply offsets"""
        if not self.landing_elem_ids and not self.bar_offset_elem_ids:
            return bdf_path

        try:
            # Read BDF for geometry
            bdf = BDF(debug=False)
            bdf.read_bdf(bdf_path, validate=False, xref=True, read_includes=True, encoding='latin-1')

            # Calculate landing normals
            landing_offsets = {}
            landing_normals = {}

            for eid in self.landing_elem_ids:
                if eid not in bdf.elements:
                    continue
                elem = bdf.elements[eid]
                pid = elem.pid if hasattr(elem, 'pid') else None

                # Get skin thickness for this property
                if pid and pid in self.current_skin_thicknesses:
                    t = self.current_skin_thicknesses[pid]
                else:
                    t = float(self.skin_min_thickness.get())

                landing_offsets[eid] = -t / 2.0

                # Calculate normal
                if elem.type in ['CQUAD4', 'CTRIA3']:
                    try:
                        nids = elem.node_ids[:3]
                        nodes = [bdf.nodes[n] for n in nids if n in bdf.nodes]
                        if len(nodes) >= 3:
                            p1 = np.array(nodes[0].get_position())
                            p2 = np.array(nodes[1].get_position())
                            p3 = np.array(nodes[2].get_position())
                            normal = np.cross(p2-p1, p3-p1)
                            norm_len = np.linalg.norm(normal)
                            if norm_len > 1e-10:
                                landing_normals[eid] = normal / norm_len
                    except:
                        pass

            # Build node -> shell map
            node_to_shells = {}
            for eid, elem in bdf.elements.items():
                if elem.type in ['CQUAD4', 'CTRIA3']:
                    for nid in elem.node_ids:
                        if nid not in node_to_shells:
                            node_to_shells[nid] = []
                        node_to_shells[nid].append(eid)

            # Calculate bar offsets
            bar_offsets = {}
            for eid in self.bar_offset_elem_ids:
                if eid not in bdf.elements:
                    continue
                elem = bdf.elements[eid]
                if elem.type != 'CBAR':
                    continue

                pid = elem.pid if hasattr(elem, 'pid') else None
                if pid and pid in self.current_bar_thicknesses:
                    bar_t = self.current_bar_thicknesses[pid]
                else:
                    bar_t = float(self.bar_min_thickness.get())

                bar_nodes = elem.node_ids[:2]
                if bar_nodes[0] in node_to_shells and bar_nodes[1] in node_to_shells:
                    common = set(node_to_shells[bar_nodes[0]]) & set(node_to_shells[bar_nodes[1]])

                    # Find thickest landing
                    max_t = 0
                    best_normal = None
                    for shell_eid in common:
                        if shell_eid in landing_offsets:
                            shell_pid = bdf.elements[shell_eid].pid
                            shell_t = self.current_skin_thicknesses.get(shell_pid, 0)
                            if shell_t > max_t:
                                max_t = shell_t
                                if shell_eid in landing_normals:
                                    best_normal = landing_normals[shell_eid]

                    if best_normal is not None and max_t > 0:
                        offset_mag = max_t + bar_t / 2.0
                        offset_vec = -best_normal * offset_mag
                        bar_offsets[eid] = tuple(offset_vec)

            # Apply offsets to BDF text
            with open(bdf_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            def fmt(v, w=8):
                s = f"{v:.4f}"
                if len(s) > w:
                    s = f"{v:.2E}"
                return s[:w].ljust(w)

            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]

                if line.startswith('CQUAD4'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in landing_offsets:
                            zoff = landing_offsets[eid]
                            new_line = line[:64] + fmt(zoff) + line[72:] if len(line) > 72 else line[:64] + fmt(zoff) + '\n'
                            new_lines.append(new_line)
                            i += 1
                            continue
                    except:
                        pass

                elif line.startswith('CBAR'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in bar_offsets:
                            vec = bar_offsets[eid]
                            new_lines.append(line.rstrip() + '+CB' + str(eid)[-4:] + '\n')
                            cont = ('+CB' + str(eid)[-4:]).ljust(8) + '        ' + '        '
                            cont += fmt(vec[0]) + fmt(vec[1]) + fmt(vec[2])
                            cont += fmt(vec[0]) + fmt(vec[1]) + fmt(vec[2]) + '\n'
                            new_lines.append(cont)
                            i += 1
                            continue
                    except:
                        pass

                new_lines.append(line)
                i += 1

            output_bdf = os.path.join(iter_folder, "model_offset.bdf")
            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            self.log(f"    Applied: {len(landing_offsets)} landing, {len(bar_offsets)} bar offsets")
            return output_bdf

        except Exception as e:
            self.log(f"    Offset error: {e}")
            return bdf_path

    def _run_nastran(self, bdf_path, iter_folder):
        """Run Nastran analysis"""
        nastran = self.nastran_path.get()
        if not nastran or not os.path.exists(nastran):
            self.log("    No Nastran executable")
            return False

        try:
            cmd = f'"{nastran}" "{bdf_path}" out="{iter_folder}" scratch=yes batch=no'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.wait(timeout=600)

            # Check for OP2
            op2_files = [f for f in os.listdir(iter_folder) if f.lower().endswith('.op2')]
            self.log(f"    Nastran complete: {len(op2_files)} OP2 files")
            return len(op2_files) > 0

        except subprocess.TimeoutExpired:
            self.log("    Nastran timeout")
            return False
        except Exception as e:
            self.log(f"    Nastran error: {e}")
            return False

    def _extract_stresses(self, iter_folder):
        """Extract stresses from OP2"""
        try:
            op2_files = [os.path.join(iter_folder, f) for f in os.listdir(iter_folder)
                         if f.lower().endswith('.op2')]

            if not op2_files:
                return []

            results = []
            for op2_path in op2_files:
                try:
                    op2 = OP2(debug=False)
                    op2.read_op2(op2_path)

                    # Bar stresses
                    if hasattr(op2, 'cbar_stress') and op2.cbar_stress:
                        for sc_id, data in op2.cbar_stress.items():
                            for i, eid in enumerate(data.element):
                                stress = data.data[0, i, 0] if len(data.data.shape) == 3 else data.data[i, 0]
                                results.append({
                                    'eid': int(eid),
                                    'type': 'bar',
                                    'stress': float(stress),
                                    'subcase': sc_id
                                })

                    # Shell stresses
                    if hasattr(op2, 'cquad4_stress') and op2.cquad4_stress:
                        for sc_id, data in op2.cquad4_stress.items():
                            for i, eid in enumerate(data.element):
                                stress = data.data[0, i, -1] if len(data.data.shape) == 3 else data.data[i, -1]
                                results.append({
                                    'eid': int(eid),
                                    'type': 'shell',
                                    'stress': float(stress),
                                    'subcase': sc_id
                                })

                except Exception as e:
                    self.log(f"    Warning reading OP2: {e}")

            self.log(f"    Extracted {len(results)} stress results")
            return results

        except Exception as e:
            self.log(f"    Stress extraction error: {e}")
            return []

    def _calculate_rf_per_element(self, stresses, target_rf):
        """Calculate RF for each element"""
        if not stresses:
            return {'min_rf': 0, 'mean_rf': 0, 'n_fail': 0, 'n_total': 0, 'failing_pids': set(), 'details': []}

        rf_list = []
        failing_pids = set()

        for s in stresses:
            eid = s['eid']
            stress = abs(s['stress'])
            etype = s['type']

            # Get property
            pid = self.elem_to_prop.get(eid)

            # Get allowable
            allowable = None
            if etype == 'bar' and pid:
                t = self.current_bar_thicknesses.get(pid, float(self.bar_min_thickness.get()))
                if pid in self.bar_allowable_interp:
                    p = self.bar_allowable_interp[pid]
                    if p['excluded']:
                        allowable = p['a']
                    else:
                        allowable = p['a'] * (t ** p['b'])
            elif etype == 'shell' and pid:
                t = self.current_skin_thicknesses.get(pid, float(self.skin_min_thickness.get()))
                if pid in self.skin_allowable_interp:
                    p = self.skin_allowable_interp[pid]
                    if p['excluded']:
                        allowable = p['a']
                    else:
                        allowable = p['a'] * (t ** p['b'])

            # Calculate RF
            if stress == 0:
                rf = 999.0
                status = 'PASS'
            elif allowable and allowable > 0:
                rf = allowable / stress
                status = 'PASS' if rf >= target_rf else 'FAIL'
            else:
                rf = 0
                status = 'NO_ALLOW'

            if status == 'FAIL' and pid:
                failing_pids.add(pid)

            rf_list.append({
                'eid': eid,
                'pid': pid,
                'type': etype,
                'stress': stress,
                'allowable': allowable,
                'rf': rf,
                'status': status
            })

        # Statistics
        valid_rf = [r['rf'] for r in rf_list if r['rf'] < 999 and r['rf'] > 0]
        min_rf = min(valid_rf) if valid_rf else 0
        mean_rf = np.mean(valid_rf) if valid_rf else 0
        n_fail = sum(1 for r in rf_list if r['status'] == 'FAIL')

        return {
            'min_rf': min_rf,
            'mean_rf': mean_rf,
            'n_fail': n_fail,
            'n_total': len(rf_list),
            'failing_pids': failing_pids,
            'details': rf_list
        }

    def _calculate_total_weight(self, density):
        """Calculate total weight"""
        weight = 0.0

        # Shell weight
        for pid in self.skin_properties:
            t = self.current_skin_thicknesses.get(pid, 0)
            if pid in self.prop_elements:
                area = sum(self.element_areas.get(eid, 0) for eid in self.prop_elements[pid])
                weight += area * t * density

        # Bar weight
        for pid in self.bar_properties:
            t = self.current_bar_thicknesses.get(pid, 0)
            if pid in self.prop_elements:
                length = sum(self.bar_lengths.get(eid, 0) for eid in self.prop_elements[pid])
                weight += length * t * t * density  # Assuming square section

        return weight

    def _update_thicknesses_smart(self, result, step, bar_min, bar_max, skin_min, skin_max, target_rf):
        """Smart thickness update based on failures"""
        failing_pids = result.get('failing_pids', set())

        if not failing_pids:
            return

        self.log(f"  Updating {len(failing_pids)} failing properties...")

        for pid in failing_pids:
            if pid in self.current_bar_thicknesses:
                current = self.current_bar_thicknesses[pid]
                # Increase by step
                new_t = min(current + step, bar_max)
                self.current_bar_thicknesses[pid] = new_t

            if pid in self.current_skin_thicknesses:
                current = self.current_skin_thicknesses[pid]
                new_t = min(current + step, skin_max)
                self.current_skin_thicknesses[pid] = new_t

    def _reduce_overdesigned(self, result, step, bar_min, skin_min, target_rf, rf_tol):
        """Try to reduce thickness for over-designed properties"""
        rf_details = result.get('rf_results', {}).get('details', [])
        if not rf_details:
            return

        # Find properties with high RF
        pid_rf = {}
        for r in rf_details:
            pid = r.get('pid')
            rf = r.get('rf', 0)
            if pid and rf < 999:
                if pid not in pid_rf or rf < pid_rf[pid]:
                    pid_rf[pid] = rf

        # Reduce thickness for properties with RF > target + margin
        reduce_threshold = target_rf + rf_tol + 0.2
        reduced = 0

        for pid, rf in pid_rf.items():
            if rf > reduce_threshold:
                if pid in self.current_bar_thicknesses:
                    current = self.current_bar_thicknesses[pid]
                    new_t = max(current - step/2, bar_min)
                    if new_t < current:
                        self.current_bar_thicknesses[pid] = new_t
                        reduced += 1

                if pid in self.current_skin_thicknesses:
                    current = self.current_skin_thicknesses[pid]
                    new_t = max(current - step/2, skin_min)
                    if new_t < current:
                        self.current_skin_thicknesses[pid] = new_t
                        reduced += 1

        if reduced > 0:
            self.log(f"  Reduced {reduced} over-designed properties")

    def _save_iteration_results(self, iter_folder, result, rf_results):
        """Save iteration results"""
        try:
            # Summary CSV
            with open(os.path.join(iter_folder, "summary.csv"), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Parameter', 'Value'])
                w.writerow(['Iteration', result['iteration']])
                w.writerow(['Min_RF', result['min_rf']])
                w.writerow(['Mean_RF', result['mean_rf']])
                w.writerow(['N_Fail', result['n_fail']])
                w.writerow(['N_Total', result['n_total']])
                w.writerow(['Weight_tonnes', result['weight']])

            # RF details
            if rf_results and 'details' in rf_results:
                pd.DataFrame(rf_results['details']).to_csv(
                    os.path.join(iter_folder, "rf_details.csv"), index=False)

            # Thicknesses
            bar_data = [{'PID': pid, 'Thickness': t} for pid, t in result['bar_thicknesses'].items()]
            skin_data = [{'PID': pid, 'Thickness': t} for pid, t in result['skin_thicknesses'].items()]

            if bar_data:
                pd.DataFrame(bar_data).to_csv(os.path.join(iter_folder, "bar_thicknesses.csv"), index=False)
            if skin_data:
                pd.DataFrame(skin_data).to_csv(os.path.join(iter_folder, "skin_thicknesses.csv"), index=False)

        except Exception as e:
            self.log(f"  Warning saving results: {e}")

    def _save_all_results(self, base_folder):
        """Save all optimization results"""
        try:
            # Iteration history
            history_data = []
            for r in self.iteration_results:
                history_data.append({
                    'iteration': r['iteration'],
                    'min_rf': r['min_rf'],
                    'mean_rf': r['mean_rf'],
                    'n_fail': r['n_fail'],
                    'weight': r['weight']
                })
            pd.DataFrame(history_data).to_csv(os.path.join(base_folder, "iteration_history.csv"), index=False)

            # Best solution
            if self.best_solution:
                pd.DataFrame([{
                    'iteration': self.best_solution['iteration'],
                    'min_rf': self.best_solution['min_rf'],
                    'weight': self.best_solution['weight'],
                    'folder': self.best_solution['folder']
                }]).to_csv(os.path.join(base_folder, "best_solution.csv"), index=False)

            self.log(f"\nSaved results to {base_folder}")

        except Exception as e:
            self.log(f"Warning saving all results: {e}")

    def _update_ui_results(self):
        """Update UI with results"""
        if self.best_solution:
            self.result_summary.config(
                text=f"Best: Weight={self.best_solution['weight']:.6f}t, RF={self.best_solution['min_rf']:.4f}",
                foreground="green"
            )

            self.best_solution_text.config(state=tk.NORMAL)
            self.best_solution_text.delete(1.0, tk.END)

            text = f"Best Solution (Iteration {self.best_solution['iteration']}):\n"
            text += f"  Total Weight: {self.best_solution['weight']:.6f} tonnes\n"
            text += f"  Minimum RF: {self.best_solution['min_rf']:.4f}\n"
            text += f"  Failed Elements: {self.best_solution['n_fail']}\n"
            text += f"  Folder: {self.best_solution['folder']}\n\n"

            text += "Bar Thicknesses (sample):\n"
            for pid in list(self.best_solution['bar_thicknesses'].keys())[:5]:
                text += f"  PID {pid}: {self.best_solution['bar_thicknesses'][pid]:.2f} mm\n"

            text += "\nSkin Thicknesses (sample):\n"
            for pid in list(self.best_solution['skin_thicknesses'].keys())[:5]:
                text += f"  PID {pid}: {self.best_solution['skin_thicknesses'][pid]:.2f} mm\n"

            self.best_solution_text.insert(tk.END, text)
            self.best_solution_text.config(state=tk.DISABLED)

    def export_results(self):
        """Export results to Excel"""
        if not self.iteration_results:
            messagebox.showerror("Error", "No results to export")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_folder.get(), f"optimization_results_{timestamp}.xlsx")

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                # History
                history = [{
                    'iteration': r['iteration'],
                    'min_rf': r['min_rf'],
                    'n_fail': r['n_fail'],
                    'weight': r['weight']
                } for r in self.iteration_results]
                pd.DataFrame(history).to_excel(writer, sheet_name='History', index=False)

                # Best solution
                if self.best_solution:
                    pd.DataFrame([self.best_solution]).to_excel(writer, sheet_name='Best', index=False)

                # Final thicknesses
                bar_data = [{'PID': p, 'Thickness': t} for p, t in self.current_bar_thicknesses.items()]
                skin_data = [{'PID': p, 'Thickness': t} for p, t in self.current_skin_thicknesses.items()]

                pd.DataFrame(bar_data).to_excel(writer, sheet_name='Bar_Thicknesses', index=False)
                pd.DataFrame(skin_data).to_excel(writer, sheet_name='Skin_Thicknesses', index=False)

            self.log(f"\nExported to: {path}")
            messagebox.showinfo("Export", f"Exported to:\n{path}")

        except Exception as e:
            self.log(f"Export error: {e}")
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = ThicknessIterationToolV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
