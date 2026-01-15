#!/usr/bin/env python3
"""
Thickness Iteration Tool v1.0
==============================
Automated thickness optimization for BDF models.

Features:
- Reads main BDF file
- Iterates bar and skin thicknesses within given ranges
- Automatically calculates and applies offsets
- Runs Nastran analysis
- Calculates RF using allowable data
- Optimizes for minimum weight while meeting RF target
- Creates separate folder for each iteration
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


class ThicknessIterationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Thickness Iteration Tool v1.0")
        self.root.geometry("1200x900")

        # Input variables
        self.input_bdf_path = tk.StringVar()
        self.allowable_excel_path = tk.StringVar()
        self.property_excel_path = tk.StringVar()
        self.element_excel_path = tk.StringVar()  # For offset calculation
        self.nastran_path = tk.StringVar()
        self.output_folder = tk.StringVar()

        # Thickness range variables
        self.bar_min_thickness = tk.StringVar(value="2.0")
        self.bar_max_thickness = tk.StringVar(value="12.0")
        self.skin_min_thickness = tk.StringVar(value="3.0")
        self.skin_max_thickness = tk.StringVar(value="18.0")
        self.thickness_step = tk.StringVar(value="0.5")

        # RF target
        self.target_rf = tk.StringVar(value="1.0")
        self.rf_tolerance = tk.StringVar(value="0.05")  # RF tolerance (e.g., 1.0 +/- 0.05)

        # Material properties
        self.density = tk.StringVar(value="2.7e-9")  # Aluminum density (tonnes/mm^3)

        # Power law fitting settings
        self.r2_threshold = tk.StringVar(value="0.95")
        self.min_data_points = tk.StringVar(value="3")

        # Optimization settings
        self.max_iterations = tk.StringVar(value="100")
        self.optimization_method = tk.StringVar(value="gradient")  # gradient, binary_search, full_scan

        # Internal data storage
        self.bar_properties = {}
        self.skin_properties = {}
        self.allowable_interp = {}  # Power law fits for bars
        self.skin_allowable_interp = {}  # Power law fits for skins
        self.residual_strength_df = None

        # BDF data
        self.bdf_model = None
        self.element_areas = {}  # Element ID -> area
        self.bar_lengths = {}  # Bar Element ID -> length
        self.prop_elements = {}  # Property ID -> list of element IDs

        # Results
        self.iteration_results = []
        self.best_solution = None
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI"""
        # Main scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = scrollable_frame

        # Title
        ttk.Label(main, text="Thickness Iteration Tool v1.0",
                  font=('Helvetica', 16, 'bold')).pack(pady=(10, 5))
        ttk.Label(main, text="Automated thickness optimization with RF targeting",
                  font=('Helvetica', 10), foreground='gray').pack(pady=(0, 10))

        # === Section 1: Input Files ===
        input_frame = ttk.LabelFrame(main, text="1. Input Files", padding="10")
        input_frame.pack(fill=tk.X, pady=5, padx=10)

        # BDF File
        row1 = ttk.Frame(input_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Main BDF File:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.input_bdf_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Browse", command=self.browse_bdf).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Load", command=self.load_bdf).pack(side=tk.LEFT, padx=2)

        self.bdf_status = ttk.Label(input_frame, text="Not loaded", foreground="gray")
        self.bdf_status.pack(anchor=tk.W, pady=2)

        # Property Excel
        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Property Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.property_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="Browse", command=self.browse_property_excel).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Load", command=self.load_properties).pack(side=tk.LEFT, padx=2)

        self.prop_status = ttk.Label(input_frame, text="Not loaded", foreground="gray")
        self.prop_status.pack(anchor=tk.W, pady=2)

        # Allowable Excel
        row3 = ttk.Frame(input_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Allowable Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.allowable_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="Browse", command=self.browse_allowable).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="Load & Fit", command=self.load_allowable).pack(side=tk.LEFT, padx=2)

        self.allow_status = ttk.Label(input_frame, text="Not loaded", foreground="gray")
        self.allow_status.pack(anchor=tk.W, pady=2)

        # Element IDs Excel (for offset)
        row4 = ttk.Frame(input_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="Element IDs Excel:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.element_excel_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row4, text="Browse", command=self.browse_element_excel).pack(side=tk.LEFT, padx=2)
        ttk.Label(row4, text="(Landing_Offset, Bar_Offset sheets)", foreground='gray').pack(side=tk.LEFT, padx=5)

        # Nastran Path
        row5 = ttk.Frame(input_frame)
        row5.pack(fill=tk.X, pady=2)
        ttk.Label(row5, text="Nastran Executable:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row5, textvariable=self.nastran_path, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row5, text="Browse", command=self.browse_nastran).pack(side=tk.LEFT, padx=2)

        # Output Folder
        row6 = ttk.Frame(input_frame)
        row6.pack(fill=tk.X, pady=2)
        ttk.Label(row6, text="Output Folder:", width=20).pack(side=tk.LEFT)
        ttk.Entry(row6, textvariable=self.output_folder, width=55).pack(side=tk.LEFT, padx=5)
        ttk.Button(row6, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=2)

        # === Section 2: Thickness Ranges ===
        range_frame = ttk.LabelFrame(main, text="2. Thickness Ranges", padding="10")
        range_frame.pack(fill=tk.X, pady=5, padx=10)

        # Bar thickness range
        bar_row = ttk.Frame(range_frame)
        bar_row.pack(fill=tk.X, pady=5)
        ttk.Label(bar_row, text="Bar Thickness:", width=15).pack(side=tk.LEFT)
        ttk.Label(bar_row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(bar_row, textvariable=self.bar_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(bar_row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(bar_row, textvariable=self.bar_max_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(bar_row, text="mm", foreground='gray').pack(side=tk.LEFT, padx=5)

        # Skin thickness range
        skin_row = ttk.Frame(range_frame)
        skin_row.pack(fill=tk.X, pady=5)
        ttk.Label(skin_row, text="Skin Thickness:", width=15).pack(side=tk.LEFT)
        ttk.Label(skin_row, text="Min:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(skin_row, textvariable=self.skin_min_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(skin_row, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(skin_row, textvariable=self.skin_max_thickness, width=8).pack(side=tk.LEFT)
        ttk.Label(skin_row, text="mm", foreground='gray').pack(side=tk.LEFT, padx=5)

        # Step size
        step_row = ttk.Frame(range_frame)
        step_row.pack(fill=tk.X, pady=5)
        ttk.Label(step_row, text="Thickness Step:", width=15).pack(side=tk.LEFT)
        ttk.Entry(step_row, textvariable=self.thickness_step, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(step_row, text="mm (for initial search)", foreground='gray').pack(side=tk.LEFT, padx=5)

        # === Section 3: RF Target ===
        rf_frame = ttk.LabelFrame(main, text="3. RF Target Settings", padding="10")
        rf_frame.pack(fill=tk.X, pady=5, padx=10)

        rf_row = ttk.Frame(rf_frame)
        rf_row.pack(fill=tk.X, pady=5)
        ttk.Label(rf_row, text="Target RF:", width=15).pack(side=tk.LEFT)
        ttk.Entry(rf_row, textvariable=self.target_rf, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(rf_row, text="Tolerance (+/-):", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(rf_row, textvariable=self.rf_tolerance, width=8).pack(side=tk.LEFT, padx=5)

        r2_row = ttk.Frame(rf_frame)
        r2_row.pack(fill=tk.X, pady=5)
        ttk.Label(r2_row, text="R² Threshold:", width=15).pack(side=tk.LEFT)
        ttk.Entry(r2_row, textvariable=self.r2_threshold, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(r2_row, text="Min Data Points:", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(r2_row, textvariable=self.min_data_points, width=8).pack(side=tk.LEFT, padx=5)

        # === Section 4: Optimization Settings ===
        opt_frame = ttk.LabelFrame(main, text="4. Optimization Settings", padding="10")
        opt_frame.pack(fill=tk.X, pady=5, padx=10)

        opt_row1 = ttk.Frame(opt_frame)
        opt_row1.pack(fill=tk.X, pady=5)
        ttk.Label(opt_row1, text="Max Iterations:", width=15).pack(side=tk.LEFT)
        ttk.Entry(opt_row1, textvariable=self.max_iterations, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(opt_row1, text="Density (t/mm³):", width=15).pack(side=tk.LEFT, padx=10)
        ttk.Entry(opt_row1, textvariable=self.density, width=12).pack(side=tk.LEFT, padx=5)

        opt_row2 = ttk.Frame(opt_frame)
        opt_row2.pack(fill=tk.X, pady=5)
        ttk.Label(opt_row2, text="Method:", width=15).pack(side=tk.LEFT)
        method_combo = ttk.Combobox(opt_row2, textvariable=self.optimization_method, width=15,
                                     values=["gradient", "binary_search", "full_scan"])
        method_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(opt_row2, text="gradient=fast convergence, binary_search=precise, full_scan=exhaustive",
                  foreground='gray').pack(side=tk.LEFT, padx=5)

        # === Section 5: Actions ===
        action_frame = ttk.LabelFrame(main, text="5. Actions", padding="10")
        action_frame.pack(fill=tk.X, pady=5, padx=10)

        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill=tk.X, pady=5)

        self.btn_start = ttk.Button(btn_row, text=">>> START ITERATION <<<",
                                     command=self.start_iteration, width=25)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(btn_row, text="STOP", command=self.stop_iteration,
                                    width=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_row, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(action_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(action_frame, text="Ready")
        self.progress_label.pack(anchor=tk.W)

        # === Section 6: Results Summary ===
        result_frame = ttk.LabelFrame(main, text="6. Results Summary", padding="10")
        result_frame.pack(fill=tk.X, pady=5, padx=10)

        self.result_summary = ttk.Label(result_frame,
                                         text="Run iteration to see results",
                                         font=('Helvetica', 11, 'bold'), foreground='blue')
        self.result_summary.pack(anchor=tk.W, pady=5)

        # Best solution details
        self.best_solution_text = tk.Text(result_frame, height=6, width=100, font=('Courier', 9))
        self.best_solution_text.pack(fill=tk.X, pady=5)
        self.best_solution_text.insert(tk.END, "Best solution will be displayed here...")
        self.best_solution_text.config(state=tk.DISABLED)

        # === Section 7: Log ===
        log_frame = ttk.LabelFrame(main, text="7. Iteration Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Initial message
        self.log("="*70)
        self.log("Thickness Iteration Tool v1.0")
        self.log("="*70)
        self.log("\nWorkflow:")
        self.log("1. Load BDF file (contains shell and bar elements)")
        self.log("2. Load Property Excel (Bar_Properties, Skin_Properties sheets)")
        self.log("3. Load Allowable Excel (Bar_Allowable, Skin_Allowable sheets)")
        self.log("4. Load Element IDs Excel (Landing_Offset, Bar_Offset sheets)")
        self.log("5. Set thickness ranges and RF target")
        self.log("6. Click 'START ITERATION'")
        self.log("\nAlgorithm:")
        self.log("- Starts with minimum thicknesses")
        self.log("- Updates properties -> calculates offsets -> runs Nastran")
        self.log("- Calculates RF from results")
        self.log("- Adjusts thicknesses based on RF sensitivity")
        self.log("- Minimizes weight while meeting RF target")
        self.log("="*70 + "\n")

    def log(self, msg):
        """Add message to log"""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear log"""
        self.log_text.delete(1.0, tk.END)

    def update_progress(self, value, text=""):
        """Update progress bar"""
        self.progress['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    # ========== FILE BROWSERS ==========
    def browse_bdf(self):
        f = filedialog.askopenfilename(filetypes=[("BDF Files", "*.bdf *.dat *.nas"), ("All", "*.*")])
        if f:
            self.input_bdf_path.set(f)

    def browse_property_excel(self):
        f = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if f:
            self.property_excel_path.set(f)

    def browse_allowable(self):
        f = filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")])
        if f:
            self.allowable_excel_path.set(f)

    def browse_element_excel(self):
        f = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if f:
            self.element_excel_path.set(f)

    def browse_nastran(self):
        f = filedialog.askopenfilename(filetypes=[("Executable", "*.exe *.bat"), ("All", "*.*")])
        if f:
            self.nastran_path.set(f)

    def browse_output(self):
        f = filedialog.askdirectory()
        if f:
            self.output_folder.set(f)

    # ========== LOADING FUNCTIONS ==========
    def load_bdf(self):
        """Load BDF file and extract element/property information"""
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

            # Count element types
            shell_count = 0
            bar_count = 0
            self.element_areas = {}
            self.bar_lengths = {}
            self.prop_elements = {}

            for eid, elem in self.bdf_model.elements.items():
                pid = elem.pid if hasattr(elem, 'pid') else None

                if pid:
                    if pid not in self.prop_elements:
                        self.prop_elements[pid] = []
                    self.prop_elements[pid].append(eid)

                if elem.type in ['CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6']:
                    shell_count += 1
                    # Calculate element area
                    try:
                        area = elem.Area()
                        self.element_areas[eid] = area
                    except:
                        self.element_areas[eid] = 0

                elif elem.type in ['CBAR', 'CBEAM']:
                    bar_count += 1
                    # Calculate bar length
                    try:
                        length = elem.Length()
                        self.bar_lengths[eid] = length
                    except:
                        self.bar_lengths[eid] = 0

            self.log(f"  Shell elements: {shell_count}")
            self.log(f"  Bar elements: {bar_count}")

            # Calculate total areas per property
            self.log("\n  Property -> Total Area/Length:")
            for pid in sorted(self.prop_elements.keys())[:10]:
                total_area = sum(self.element_areas.get(eid, 0) for eid in self.prop_elements[pid])
                total_length = sum(self.bar_lengths.get(eid, 0) for eid in self.prop_elements[pid])
                if total_area > 0:
                    self.log(f"    PID {pid}: Area = {total_area:.2f} mm²")
                if total_length > 0:
                    self.log(f"    PID {pid}: Length = {total_length:.2f} mm")

            self.bdf_status.config(
                text=f"✓ Loaded: {n_elements} elements, {n_properties} properties",
                foreground="green"
            )

            # Set output folder if not set
            if not self.output_folder.get():
                self.output_folder.set(os.path.dirname(path))

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.bdf_status.config(text=f"Error: {e}", foreground="red")
            messagebox.showerror("Error", str(e))

    def load_properties(self):
        """Load property definitions from Excel"""
        path = self.property_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select a valid Property Excel file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING PROPERTY DEFINITIONS")
        self.log("="*70)

        try:
            xl = pd.ExcelFile(path)
            sheets = xl.sheet_names
            self.log(f"Available sheets: {', '.join(sheets)}")

            # Load Bar Properties
            bar_sheet = None
            skin_sheet = None
            residual_sheet = None

            for s in sheets:
                s_lower = s.lower().replace('_', '').replace(' ', '')
                if 'bar' in s_lower and 'prop' in s_lower:
                    bar_sheet = s
                elif 'skin' in s_lower and 'prop' in s_lower:
                    skin_sheet = s
                elif 'residual' in s_lower or 'strength' in s_lower:
                    residual_sheet = s

            self.bar_properties = {}
            self.skin_properties = {}

            if bar_sheet:
                self.log(f"\nReading '{bar_sheet}'...")
                df = pd.read_excel(xl, sheet_name=bar_sheet)
                for _, row in df.iterrows():
                    pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                    if pid:
                        self.bar_properties[pid] = {
                            'dim1': float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else 0,
                            'dim2': float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 0,
                            'type': str(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else 'BOX'
                        }
                self.log(f"  Loaded {len(self.bar_properties)} bar properties")

            if skin_sheet:
                self.log(f"\nReading '{skin_sheet}'...")
                df = pd.read_excel(xl, sheet_name=skin_sheet)
                for _, row in df.iterrows():
                    pid = int(row.iloc[0]) if pd.notna(row.iloc[0]) else None
                    if pid:
                        self.skin_properties[pid] = {
                            'thickness': float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else 0,
                            'material': int(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 1
                        }
                self.log(f"  Loaded {len(self.skin_properties)} skin properties")

            if residual_sheet:
                self.log(f"\nReading '{residual_sheet}'...")
                self.residual_strength_df = pd.read_excel(xl, sheet_name=residual_sheet)
                self.log(f"  Loaded residual strength data: {len(self.residual_strength_df)} rows")

            self.prop_status.config(
                text=f"✓ Bar: {len(self.bar_properties)}, Skin: {len(self.skin_properties)}",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.prop_status.config(text=f"Error: {e}", foreground="red")
            messagebox.showerror("Error", str(e))

    def load_allowable(self):
        """Load allowable data and fit power law curves"""
        path = self.allowable_excel_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select a valid Allowable file")
            return

        self.log("\n" + "="*70)
        self.log("LOADING ALLOWABLE DATA & FITTING POWER LAW")
        self.log("="*70)

        try:
            r2_threshold = float(self.r2_threshold.get())
            min_data_pts = int(self.min_data_points.get())
        except:
            r2_threshold = 0.95
            min_data_pts = 3

        self.log(f"R² Threshold: {r2_threshold}, Min Data Points: {min_data_pts}")

        try:
            if path.endswith('.csv'):
                raw_df = pd.read_csv(path)
                # Try to detect bar vs skin from columns
                self._fit_allowable_curves(raw_df, 'bar', r2_threshold, min_data_pts)
            else:
                xl = pd.ExcelFile(path)
                sheets = xl.sheet_names
                self.log(f"Available sheets: {', '.join(sheets)}")

                bar_sheet = None
                skin_sheet = None

                for s in sheets:
                    s_lower = s.lower().replace('_', '').replace(' ', '')
                    if 'bar' in s_lower and 'allow' in s_lower:
                        bar_sheet = s
                    elif 'skin' in s_lower and 'allow' in s_lower:
                        skin_sheet = s

                if bar_sheet:
                    self.log(f"\nProcessing '{bar_sheet}'...")
                    df = pd.read_excel(xl, sheet_name=bar_sheet)
                    self._fit_allowable_curves(df, 'bar', r2_threshold, min_data_pts)

                if skin_sheet:
                    self.log(f"\nProcessing '{skin_sheet}'...")
                    df = pd.read_excel(xl, sheet_name=skin_sheet)
                    self._fit_allowable_curves(df, 'skin', r2_threshold, min_data_pts)

            total_bar = len(self.allowable_interp)
            total_skin = len(self.skin_allowable_interp)

            self.allow_status.config(
                text=f"✓ Bar: {total_bar} fits, Skin: {total_skin} fits",
                foreground="green"
            )

        except Exception as e:
            self.log(f"\nERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.allow_status.config(text=f"Error: {e}", foreground="red")
            messagebox.showerror("Error", str(e))

    def _fit_allowable_curves(self, df, prop_type, r2_threshold, min_data_pts):
        """Fit power law curves for allowable data"""
        # Map column names
        col_map = {}
        for col in df.columns:
            col_up = col.upper().replace(' ', '_').strip('_')

            if col_up in ['PROPERTY_ID', 'PROPERTY', 'PROP_ID', 'PID']:
                col_map[col] = 'Property'
            elif col_up in ['T', 'THICKNESS', 'T_MM', 'DIM1']:
                col_map[col] = 'Thickness'
            elif col_up in ['ALLOWABLE', 'ALLOW', 'ALLOWABLE_STRESS', 'ALLOWABLE_MPa']:
                col_map[col] = 'Allowable'

        df = df.rename(columns=col_map)

        # Convert to numeric
        df['Thickness'] = pd.to_numeric(df['Thickness'], errors='coerce')
        df['Allowable'] = pd.to_numeric(df['Allowable'], errors='coerce')
        df['Property'] = pd.to_numeric(df['Property'], errors='coerce')
        df = df.dropna(subset=['Property', 'Thickness', 'Allowable'])

        properties = df['Property'].unique()
        self.log(f"  Fitting {len(properties)} properties...")

        interp_dict = self.allowable_interp if prop_type == 'bar' else self.skin_allowable_interp
        valid_props = []
        excluded_r2 = []
        excluded_data = []

        for pid in properties:
            pid_int = int(pid)
            prop_data = df[df['Property'] == pid]
            n_pts = len(prop_data)

            if n_pts < min_data_pts:
                avg = prop_data['Allowable'].mean()
                interp_dict[pid_int] = {'a': avg, 'b': 0, 'n_pts': n_pts, 'r2': 0, 'excluded': True}
                excluded_data.append((pid_int, n_pts))
                continue

            try:
                x = prop_data['Thickness'].values.astype(float)
                y = prop_data['Allowable'].values.astype(float)

                valid = (x > 0) & (y > 0)
                x, y = x[valid], y[valid]

                if len(x) < 2:
                    interp_dict[pid_int] = {'a': np.mean(y), 'b': 0, 'n_pts': len(x), 'r2': 0, 'excluded': True}
                    excluded_data.append((pid_int, len(x)))
                    continue

                # Power law fit: Allowable = a * T^b
                log_x, log_y = np.log(x), np.log(y)
                coeffs = np.polyfit(log_x, log_y, 1)
                b, log_a = coeffs[0], coeffs[1]
                a = np.exp(log_a)

                # R²
                y_pred = a * (x ** b)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r2 < r2_threshold:
                    interp_dict[pid_int] = {'a': np.mean(y), 'b': 0, 'n_pts': n_pts, 'r2': r2, 'excluded': True}
                    excluded_r2.append((pid_int, r2, n_pts))
                else:
                    interp_dict[pid_int] = {'a': a, 'b': b, 'n_pts': n_pts, 'r2': r2, 'excluded': False}
                    valid_props.append(pid_int)

            except Exception as e:
                interp_dict[pid_int] = {'a': 100, 'b': 0, 'n_pts': n_pts, 'r2': 0, 'excluded': True}
                excluded_data.append((pid_int, n_pts))

        # Report
        self.log(f"\n  Valid fits (R² >= {r2_threshold}): {len(valid_props)}")
        self.log(f"  Excluded (R² < {r2_threshold}): {len(excluded_r2)}")
        self.log(f"  Excluded (data < {min_data_pts}): {len(excluded_data)}")

        if valid_props:
            self.log(f"\n  Sample valid fits:")
            for pid in valid_props[:5]:
                p = interp_dict[pid]
                self.log(f"    Property {pid}: Allow = {p['a']:.4f} × T^({p['b']:.4f}), R²={p['r2']:.4f}")

    # ========== ITERATION FUNCTIONS ==========
    def start_iteration(self):
        """Start the thickness iteration process"""
        # Validate inputs
        if not self.input_bdf_path.get():
            messagebox.showerror("Error", "Load BDF file first")
            return
        if not self.bdf_model:
            messagebox.showerror("Error", "Load BDF file first")
            return
        if not self.allowable_interp and not self.skin_allowable_interp:
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

        threading.Thread(target=self.run_iteration, daemon=True).start()

    def stop_iteration(self):
        """Stop the iteration process"""
        self.is_running = False
        self.log("\n*** STOPPING ITERATION ***")

    def run_iteration(self):
        """Main iteration loop"""
        try:
            self.log("\n" + "="*70)
            self.log("STARTING THICKNESS ITERATION")
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
            density = float(self.density.get())
            method = self.optimization_method.get()

            self.log(f"\nParameters:")
            self.log(f"  Bar thickness: {bar_min} - {bar_max} mm")
            self.log(f"  Skin thickness: {skin_min} - {skin_max} mm")
            self.log(f"  Step: {step} mm")
            self.log(f"  Target RF: {target_rf} +/- {rf_tol}")
            self.log(f"  Max iterations: {max_iter}")
            self.log(f"  Density: {density} t/mm³")
            self.log(f"  Method: {method}")

            # Create base output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_folder = os.path.join(self.output_folder.get(), f"iteration_{timestamp}")
            os.makedirs(base_folder, exist_ok=True)

            self.log(f"\nOutput folder: {base_folder}")

            # Initialize with minimum thicknesses
            current_bar_t = bar_min
            current_skin_t = skin_min

            iteration = 0
            converged = False
            best_weight = float('inf')
            best_rf = 0
            best_bar_t = bar_min
            best_skin_t = skin_min

            # History for gradient calculation
            history = []

            while iteration < max_iter and self.is_running and not converged:
                iteration += 1
                self.log(f"\n{'='*60}")
                self.log(f"ITERATION {iteration}")
                self.log(f"{'='*60}")
                self.log(f"  Bar thickness: {current_bar_t:.2f} mm")
                self.log(f"  Skin thickness: {current_skin_t:.2f} mm")

                # Update progress
                progress = (iteration / max_iter) * 100
                self.update_progress(progress, f"Iteration {iteration}/{max_iter}")

                # Create iteration folder
                iter_folder = os.path.join(base_folder, f"iter_{iteration:03d}")
                os.makedirs(iter_folder, exist_ok=True)

                # === Step 1: Update properties in BDF ===
                self.log("\n  [1] Updating properties...")
                bdf_path = self.update_bdf_properties(iter_folder, current_bar_t, current_skin_t)

                if not bdf_path:
                    self.log("  ERROR: Failed to update BDF")
                    continue

                # === Step 2: Calculate and apply offsets ===
                self.log("  [2] Calculating and applying offsets...")
                offset_bdf_path = self.calculate_and_apply_offsets(bdf_path, iter_folder,
                                                                    current_bar_t, current_skin_t)

                if not offset_bdf_path:
                    offset_bdf_path = bdf_path  # Use non-offset version if offset fails

                # === Step 3: Run Nastran ===
                self.log("  [3] Running Nastran...")
                run_success = self.run_nastran(offset_bdf_path, iter_folder)

                if not run_success:
                    self.log("  WARNING: Nastran run failed, skipping this iteration")
                    continue

                # === Step 4: Extract stress results ===
                self.log("  [4] Extracting stress results...")
                stress_results = self.extract_stress_results(iter_folder)

                if not stress_results:
                    self.log("  WARNING: No stress results, skipping this iteration")
                    continue

                # === Step 5: Calculate RF ===
                self.log("  [5] Calculating RF...")
                rf_results = self.calculate_rf(stress_results, current_bar_t, current_skin_t)

                if not rf_results:
                    self.log("  WARNING: RF calculation failed")
                    continue

                min_rf = rf_results['min_rf']
                n_fail = rf_results['n_fail']

                # === Step 6: Calculate weight ===
                self.log("  [6] Calculating weight...")
                weight = self.calculate_weight(current_bar_t, current_skin_t, density)

                self.log(f"\n  Results:")
                self.log(f"    Min RF: {min_rf:.4f}")
                self.log(f"    Failed elements: {n_fail}")
                self.log(f"    Total weight: {weight:.4f} tonnes")

                # Save iteration results
                iter_result = {
                    'iteration': iteration,
                    'bar_thickness': current_bar_t,
                    'skin_thickness': current_skin_t,
                    'min_rf': min_rf,
                    'n_fail': n_fail,
                    'weight': weight,
                    'folder': iter_folder
                }
                self.iteration_results.append(iter_result)
                history.append(iter_result)

                # Save iteration summary
                self.save_iteration_summary(iter_folder, iter_result, rf_results)

                # Check if this is the best solution
                rf_in_range = (target_rf - rf_tol) <= min_rf <= (target_rf + rf_tol + 0.5)  # Allow some overshoot

                if rf_in_range and weight < best_weight:
                    best_weight = weight
                    best_rf = min_rf
                    best_bar_t = current_bar_t
                    best_skin_t = current_skin_t
                    self.best_solution = iter_result.copy()
                    self.log(f"\n  *** NEW BEST SOLUTION ***")
                    self.log(f"    Weight: {best_weight:.4f} tonnes")
                    self.log(f"    RF: {best_rf:.4f}")

                # Check convergence
                if abs(min_rf - target_rf) < rf_tol and n_fail == 0:
                    self.log(f"\n  *** CONVERGED ***")
                    converged = True
                    break

                # === Step 7: Adjust thicknesses for next iteration ===
                self.log("  [7] Calculating next thickness values...")

                if method == "gradient":
                    current_bar_t, current_skin_t = self.gradient_update(
                        history, current_bar_t, current_skin_t,
                        target_rf, bar_min, bar_max, skin_min, skin_max, step
                    )
                elif method == "binary_search":
                    current_bar_t, current_skin_t = self.binary_search_update(
                        history, current_bar_t, current_skin_t,
                        target_rf, bar_min, bar_max, skin_min, skin_max
                    )
                else:  # full_scan
                    current_bar_t, current_skin_t = self.full_scan_update(
                        current_bar_t, current_skin_t,
                        bar_min, bar_max, skin_min, skin_max, step
                    )

                # Ensure within bounds
                current_bar_t = max(bar_min, min(bar_max, current_bar_t))
                current_skin_t = max(skin_min, min(skin_max, current_skin_t))

            # === Final Summary ===
            self.log("\n" + "="*70)
            self.log("ITERATION COMPLETE")
            self.log("="*70)

            if self.best_solution:
                self.log(f"\nBest Solution Found:")
                self.log(f"  Bar thickness: {self.best_solution['bar_thickness']:.2f} mm")
                self.log(f"  Skin thickness: {self.best_solution['skin_thickness']:.2f} mm")
                self.log(f"  Min RF: {self.best_solution['min_rf']:.4f}")
                self.log(f"  Weight: {self.best_solution['weight']:.4f} tonnes")
                self.log(f"  Folder: {self.best_solution['folder']}")

                # Update UI
                self.root.after(0, lambda: self.update_best_solution_display())
            else:
                self.log("\nNo valid solution found within constraints.")

            # Save all results
            self.save_all_results(base_folder)

            self.log(f"\nTotal iterations: {iteration}")
            self.log(f"Results saved to: {base_folder}")

            self.root.after(0, lambda: messagebox.showinfo("Complete",
                f"Iteration complete!\n\nTotal iterations: {iteration}\nResults: {base_folder}"))

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

    def update_bdf_properties(self, iter_folder, bar_t, skin_t):
        """Update BDF properties with new thicknesses"""
        try:
            # Copy original BDF
            input_bdf = self.input_bdf_path.get()
            output_bdf = os.path.join(iter_folder, "model_updated.bdf")

            # Read as text to preserve structure
            with open(input_bdf, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            new_lines = []
            bar_updated = 0
            skin_updated = 0

            i = 0
            while i < len(lines):
                line = lines[i]

                # Update PBARL
                if line.startswith('PBARL'):
                    try:
                        pid = int(line[8:16].strip())
                        if pid in self.bar_properties:
                            # Update dimensions (dim1 = dim2 = bar_t for square section)
                            # Keep first two lines, update dimensions in continuation
                            new_lines.append(line)
                            i += 1

                            # Look for continuation line with dimensions
                            while i < len(lines) and (lines[i].startswith('+') or lines[i].startswith('*') or lines[i].startswith(' ')):
                                cont_line = lines[i]
                                if cont_line.strip() and not cont_line.strip().startswith('$'):
                                    # This is a continuation line, update dimensions
                                    # Format: +name   DIM1    DIM2    ...
                                    new_cont = cont_line[:8]  # Keep continuation marker
                                    new_cont += f"{bar_t:8.4f}"  # DIM1
                                    new_cont += f"{bar_t:8.4f}"  # DIM2
                                    if len(cont_line) > 24:
                                        new_cont += cont_line[24:]
                                    else:
                                        new_cont += '\n'
                                    new_lines.append(new_cont)
                                    bar_updated += 1
                                else:
                                    new_lines.append(cont_line)
                                i += 1
                            continue
                    except:
                        pass
                    new_lines.append(line)
                    i += 1
                    continue

                # Update PSHELL
                elif line.startswith('PSHELL'):
                    try:
                        pid = int(line[8:16].strip())
                        if pid in self.skin_properties:
                            # Update thickness in field 4 (columns 24-32)
                            new_line = line[:24] + f"{skin_t:8.4f}" + line[32:]
                            new_lines.append(new_line)
                            skin_updated += 1
                            i += 1
                            continue
                    except:
                        pass
                    new_lines.append(line)
                    i += 1
                    continue

                else:
                    new_lines.append(line)
                    i += 1

            # Write updated BDF
            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            self.log(f"    Updated {bar_updated} PBARL, {skin_updated} PSHELL")
            return output_bdf

        except Exception as e:
            self.log(f"    ERROR updating BDF: {e}")
            return None

    def calculate_and_apply_offsets(self, bdf_path, iter_folder, bar_t, skin_t):
        """Calculate and apply offsets to BDF"""
        try:
            # Read element IDs from Excel
            element_excel = self.element_excel_path.get()
            if not element_excel or not os.path.exists(element_excel):
                self.log("    No element Excel, skipping offsets")
                return bdf_path

            xl = pd.ExcelFile(element_excel)
            sheets = xl.sheet_names

            landing_sheet = bar_sheet = None
            for s in sheets:
                s_lower = s.lower().replace('_', '').replace(' ', '')
                if 'landing' in s_lower and 'offset' in s_lower:
                    landing_sheet = s
                elif 'bar' in s_lower and 'offset' in s_lower:
                    bar_sheet = s

            landing_elem_ids = []
            bar_elem_ids = []

            if landing_sheet:
                df = pd.read_excel(xl, sheet_name=landing_sheet)
                landing_elem_ids = df.iloc[:,0].dropna().astype(int).tolist()

            if bar_sheet:
                df = pd.read_excel(xl, sheet_name=bar_sheet)
                bar_elem_ids = df.iloc[:,0].dropna().astype(int).tolist()

            if not landing_elem_ids and not bar_elem_ids:
                self.log("    No element IDs found, skipping offsets")
                return bdf_path

            # Read BDF with pyNastran
            bdf = BDF(debug=False)
            bdf.read_bdf(bdf_path, validate=False, xref=True, read_includes=True, encoding='latin-1')

            # Calculate landing offsets
            landing_offsets = {}
            landing_normals = {}
            landing_thickness = {}

            for eid in landing_elem_ids:
                if eid in bdf.elements:
                    elem = bdf.elements[eid]
                    # Use current skin thickness
                    thickness = skin_t
                    zoffset = -thickness / 2.0
                    landing_offsets[eid] = zoffset
                    landing_thickness[eid] = thickness

                    # Calculate normal
                    if elem.type in ['CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6']:
                        node_ids = elem.node_ids[:4] if elem.type.startswith('CQUAD') else elem.node_ids[:3]
                        nodes = [bdf.nodes[nid] for nid in node_ids if nid in bdf.nodes]

                        if len(nodes) >= 3:
                            p1 = np.array(nodes[0].get_position())
                            p2 = np.array(nodes[1].get_position())
                            p3 = np.array(nodes[2].get_position())

                            v1 = p2 - p1
                            v2 = p3 - p1
                            normal = np.cross(v1, v2)
                            normal_len = np.linalg.norm(normal)

                            if normal_len > 1e-10:
                                landing_normals[eid] = normal / normal_len

            # Build node -> shell mapping
            node_to_shells = {}
            for eid, elem in bdf.elements.items():
                if elem.type in ['CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6']:
                    for nid in elem.node_ids:
                        if nid not in node_to_shells:
                            node_to_shells[nid] = []
                        node_to_shells[nid].append(eid)

            # Calculate bar offsets
            bar_offsets = {}

            for eid in bar_elem_ids:
                if eid in bdf.elements:
                    elem = bdf.elements[eid]
                    if elem.type == 'CBAR':
                        bar_nodes = elem.node_ids[:2]

                        if bar_nodes[0] in node_to_shells and bar_nodes[1] in node_to_shells:
                            shells_n1 = set(node_to_shells[bar_nodes[0]])
                            shells_n2 = set(node_to_shells[bar_nodes[1]])
                            connected_shells = shells_n1.intersection(shells_n2)

                            max_landing_thick = 0
                            landing_normal = None

                            for shell_eid in connected_shells:
                                if shell_eid in landing_thickness:
                                    t = landing_thickness[shell_eid]
                                    if t > max_landing_thick:
                                        max_landing_thick = t
                                        if shell_eid in landing_normals:
                                            landing_normal = landing_normals[shell_eid]

                            if landing_normal is not None and max_landing_thick > 0:
                                offset_magnitude = max_landing_thick + (bar_t / 2.0)
                                offset_vector = -landing_normal * offset_magnitude
                                bar_offsets[eid] = (offset_vector[0], offset_vector[1], offset_vector[2])

            self.log(f"    Landing offsets: {len(landing_offsets)}, Bar offsets: {len(bar_offsets)}")

            # Apply offsets to BDF
            with open(bdf_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

            def fmt_field(value, width=8):
                s = f"{value:.4f}"
                if len(s) > width:
                    s = f"{value:.2E}"
                return s[:width].ljust(width)

            new_lines = []
            i = 0
            landing_modified = 0
            bar_modified = 0

            while i < len(lines):
                line = lines[i]

                # CQUAD4 landing offset
                if line.startswith('CQUAD4'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in landing_offsets:
                            zoff = landing_offsets[eid]
                            if len(line) >= 64:
                                new_line = line[:64] + fmt_field(zoff)
                                if len(line) > 72:
                                    new_line += line[72:]
                                else:
                                    new_line += '\n'
                                new_lines.append(new_line)
                                landing_modified += 1
                                i += 1
                                continue
                    except:
                        pass
                    new_lines.append(line)
                    i += 1
                    continue

                # CBAR offset
                elif line.startswith('CBAR'):
                    try:
                        eid = int(line[8:16].strip())
                        if eid in bar_offsets:
                            offset_vec = bar_offsets[eid]

                            # Add continuation line for offsets
                            new_lines.append(line.rstrip() + '+CB' + str(eid)[-4:] + '\n')
                            cont_name = '+CB' + str(eid)[-4:]
                            new_cont = cont_name.ljust(8) + '        ' + '        '
                            new_cont += fmt_field(offset_vec[0])
                            new_cont += fmt_field(offset_vec[1])
                            new_cont += fmt_field(offset_vec[2])
                            new_cont += fmt_field(offset_vec[0])
                            new_cont += fmt_field(offset_vec[1])
                            new_cont += fmt_field(offset_vec[2])
                            new_cont += '\n'
                            new_lines.append(new_cont)
                            bar_modified += 1
                            i += 1
                            continue
                    except:
                        pass
                    new_lines.append(line)
                    i += 1
                    continue

                else:
                    new_lines.append(line)
                    i += 1

            # Write offset BDF
            output_bdf = os.path.join(iter_folder, "model_offset.bdf")
            with open(output_bdf, 'w', encoding='latin-1') as f:
                f.writelines(new_lines)

            self.log(f"    Applied: {landing_modified} landing, {bar_modified} bar offsets")
            return output_bdf

        except Exception as e:
            self.log(f"    ERROR applying offsets: {e}")
            return bdf_path

    def run_nastran(self, bdf_path, iter_folder):
        """Run Nastran analysis"""
        try:
            nastran = self.nastran_path.get()
            if not nastran or not os.path.exists(nastran):
                self.log("    No Nastran path, skipping analysis")
                return False

            cmd = f'"{nastran}" "{bdf_path}" out="{iter_folder}" scratch=yes batch=no'

            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait(timeout=600)  # 10 minute timeout

            # Check for OP2 file
            op2_files = [f for f in os.listdir(iter_folder) if f.lower().endswith('.op2')]

            if op2_files:
                self.log(f"    Nastran completed, {len(op2_files)} OP2 files")
                return True
            else:
                self.log("    WARNING: No OP2 files generated")
                return False

        except subprocess.TimeoutExpired:
            self.log("    WARNING: Nastran timeout")
            return False
        except Exception as e:
            self.log(f"    ERROR running Nastran: {e}")
            return False

    def extract_stress_results(self, iter_folder):
        """Extract stress results from OP2 files"""
        try:
            op2_files = [os.path.join(iter_folder, f) for f in os.listdir(iter_folder)
                         if f.lower().endswith('.op2')]

            if not op2_files:
                return None

            results = []

            for op2_path in op2_files:
                try:
                    op2 = OP2(debug=False)
                    op2.read_op2(op2_path)

                    # Extract bar stresses
                    if hasattr(op2, 'cbar_stress') and op2.cbar_stress:
                        for sc_id, stress_data in op2.cbar_stress.items():
                            for i, eid in enumerate(stress_data.element):
                                # Get axial stress
                                if len(stress_data.data.shape) == 3:
                                    axial = stress_data.data[0, i, 0]  # Axial stress
                                else:
                                    axial = stress_data.data[i, 0]

                                results.append({
                                    'type': 'bar',
                                    'element': eid,
                                    'subcase': sc_id,
                                    'stress': axial
                                })

                    # Extract shell stresses
                    if hasattr(op2, 'cquad4_stress') and op2.cquad4_stress:
                        for sc_id, stress_data in op2.cquad4_stress.items():
                            for i, eid in enumerate(stress_data.element):
                                # Get von Mises stress
                                if len(stress_data.data.shape) == 3:
                                    vm = stress_data.data[0, i, -1]  # von Mises
                                else:
                                    vm = stress_data.data[i, -1]

                                results.append({
                                    'type': 'shell',
                                    'element': eid,
                                    'subcase': sc_id,
                                    'stress': vm
                                })

                except Exception as e:
                    self.log(f"    Warning: Error reading {os.path.basename(op2_path)}: {e}")

            self.log(f"    Extracted {len(results)} stress results")
            return results

        except Exception as e:
            self.log(f"    ERROR extracting results: {e}")
            return None

    def calculate_rf(self, stress_results, bar_t, skin_t):
        """Calculate Reserve Factors"""
        try:
            rf_list = []
            n_fail = 0
            target_rf = float(self.target_rf.get())

            for result in stress_results:
                eid = result['element']
                stress = abs(result['stress'])
                elem_type = result['type']

                if stress == 0:
                    rf = 999.0
                    status = 'PASS'
                else:
                    # Get allowable
                    allowable = None

                    if elem_type == 'bar':
                        # Use bar thickness
                        for pid in self.bar_properties:
                            if pid in self.allowable_interp:
                                params = self.allowable_interp[pid]
                                if params.get('excluded', False):
                                    allowable = params['a']
                                else:
                                    allowable = params['a'] * (bar_t ** params['b'])
                                break
                    else:  # shell
                        for pid in self.skin_properties:
                            if pid in self.skin_allowable_interp:
                                params = self.skin_allowable_interp[pid]
                                if params.get('excluded', False):
                                    allowable = params['a']
                                else:
                                    allowable = params['a'] * (skin_t ** params['b'])
                                break

                    if allowable and allowable > 0:
                        rf = allowable / stress
                    else:
                        rf = 0.0

                    status = 'PASS' if rf >= target_rf else 'FAIL'
                    if status == 'FAIL':
                        n_fail += 1

                rf_list.append({
                    'element': eid,
                    'type': elem_type,
                    'stress': stress,
                    'rf': rf,
                    'status': status
                })

            if rf_list:
                min_rf = min(r['rf'] for r in rf_list if r['rf'] < 999)
                mean_rf = np.mean([r['rf'] for r in rf_list if r['rf'] < 999])
            else:
                min_rf = 0
                mean_rf = 0

            return {
                'rf_list': rf_list,
                'min_rf': min_rf,
                'mean_rf': mean_rf,
                'n_fail': n_fail,
                'n_total': len(rf_list)
            }

        except Exception as e:
            self.log(f"    ERROR calculating RF: {e}")
            return None

    def calculate_weight(self, bar_t, skin_t, density):
        """Calculate total weight"""
        try:
            total_weight = 0.0

            # Skin weight: area * thickness * density
            for pid in self.skin_properties:
                if pid in self.prop_elements:
                    total_area = sum(self.element_areas.get(eid, 0) for eid in self.prop_elements[pid])
                    skin_weight = total_area * skin_t * density
                    total_weight += skin_weight

            # Bar weight: length * dim1 * dim2 * density (assuming square section)
            for pid in self.bar_properties:
                if pid in self.prop_elements:
                    total_length = sum(self.bar_lengths.get(eid, 0) for eid in self.prop_elements[pid])
                    bar_weight = total_length * bar_t * bar_t * density
                    total_weight += bar_weight

            return total_weight

        except Exception as e:
            self.log(f"    ERROR calculating weight: {e}")
            return 0.0

    def gradient_update(self, history, bar_t, skin_t, target_rf, bar_min, bar_max, skin_min, skin_max, step):
        """Update thicknesses using gradient-based approach"""
        if len(history) < 2:
            # Not enough data, increase thickness
            return bar_t + step, skin_t + step

        # Get last two results
        last = history[-1]
        prev = history[-2]

        # Calculate RF gradient with respect to thickness
        d_bar = last['bar_thickness'] - prev['bar_thickness']
        d_skin = last['skin_thickness'] - prev['skin_thickness']
        d_rf = last['min_rf'] - prev['min_rf']

        # Error from target
        rf_error = target_rf - last['min_rf']

        # Update based on gradient
        learning_rate = 0.5

        if abs(d_bar) > 0.001:
            grad_bar = d_rf / d_bar
            bar_update = learning_rate * rf_error / (grad_bar if abs(grad_bar) > 0.001 else 0.1)
            bar_t = bar_t + np.clip(bar_update, -step*2, step*2)
        elif rf_error > 0:
            bar_t = bar_t + step

        if abs(d_skin) > 0.001:
            grad_skin = d_rf / d_skin
            skin_update = learning_rate * rf_error / (grad_skin if abs(grad_skin) > 0.001 else 0.1)
            skin_t = skin_t + np.clip(skin_update, -step*2, step*2)
        elif rf_error > 0:
            skin_t = skin_t + step

        return bar_t, skin_t

    def binary_search_update(self, history, bar_t, skin_t, target_rf, bar_min, bar_max, skin_min, skin_max):
        """Update thicknesses using binary search"""
        last = history[-1]

        if last['min_rf'] < target_rf:
            # RF too low, increase thickness
            bar_t = (bar_t + bar_max) / 2
            skin_t = (skin_t + skin_max) / 2
        else:
            # RF ok, try to reduce thickness
            bar_t = (bar_t + bar_min) / 2
            skin_t = (skin_t + skin_min) / 2

        return bar_t, skin_t

    def full_scan_update(self, bar_t, skin_t, bar_min, bar_max, skin_min, skin_max, step):
        """Full scan - just increment"""
        bar_t = bar_t + step
        if bar_t > bar_max:
            bar_t = bar_min
            skin_t = skin_t + step

        return bar_t, skin_t

    def save_iteration_summary(self, iter_folder, iter_result, rf_results):
        """Save iteration summary to folder"""
        try:
            summary_path = os.path.join(iter_folder, "iteration_summary.csv")

            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(['Iteration', iter_result['iteration']])
                writer.writerow(['Bar_Thickness', iter_result['bar_thickness']])
                writer.writerow(['Skin_Thickness', iter_result['skin_thickness']])
                writer.writerow(['Min_RF', iter_result['min_rf']])
                writer.writerow(['N_Fail', iter_result['n_fail']])
                writer.writerow(['Weight', iter_result['weight']])

            # Save RF details
            if rf_results and 'rf_list' in rf_results:
                rf_path = os.path.join(iter_folder, "rf_results.csv")
                pd.DataFrame(rf_results['rf_list']).to_csv(rf_path, index=False)

        except Exception as e:
            self.log(f"    Warning: Could not save summary: {e}")

    def save_all_results(self, base_folder):
        """Save all iteration results"""
        try:
            # Save iteration history
            history_path = os.path.join(base_folder, "iteration_history.csv")
            pd.DataFrame(self.iteration_results).to_csv(history_path, index=False)

            # Save best solution
            if self.best_solution:
                best_path = os.path.join(base_folder, "best_solution.csv")
                pd.DataFrame([self.best_solution]).to_csv(best_path, index=False)

            self.log(f"\n  Saved iteration_history.csv")
            self.log(f"  Saved best_solution.csv")

        except Exception as e:
            self.log(f"    Warning: Could not save all results: {e}")

    def update_best_solution_display(self):
        """Update the best solution display in UI"""
        if self.best_solution:
            self.result_summary.config(
                text=f"Best: Bar={self.best_solution['bar_thickness']:.2f}mm, "
                     f"Skin={self.best_solution['skin_thickness']:.2f}mm, "
                     f"RF={self.best_solution['min_rf']:.4f}, "
                     f"Weight={self.best_solution['weight']:.4f}t",
                foreground="green"
            )

            self.best_solution_text.config(state=tk.NORMAL)
            self.best_solution_text.delete(1.0, tk.END)
            self.best_solution_text.insert(tk.END,
                f"Best Solution Found:\n"
                f"  Iteration: {self.best_solution['iteration']}\n"
                f"  Bar Thickness: {self.best_solution['bar_thickness']:.4f} mm\n"
                f"  Skin Thickness: {self.best_solution['skin_thickness']:.4f} mm\n"
                f"  Minimum RF: {self.best_solution['min_rf']:.4f}\n"
                f"  Total Weight: {self.best_solution['weight']:.6f} tonnes\n"
                f"  Result Folder: {self.best_solution['folder']}"
            )
            self.best_solution_text.config(state=tk.DISABLED)

    def export_results(self):
        """Export all results to Excel"""
        if not self.iteration_results:
            messagebox.showerror("Error", "No results to export")
            return

        try:
            out_dir = self.output_folder.get()
            if not out_dir:
                out_dir = filedialog.askdirectory(title="Select Output Folder")
                if not out_dir:
                    return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join(out_dir, f"iteration_results_{timestamp}.xlsx")

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Iteration history
                pd.DataFrame(self.iteration_results).to_excel(writer, sheet_name='History', index=False)

                # Best solution
                if self.best_solution:
                    pd.DataFrame([self.best_solution]).to_excel(writer, sheet_name='Best_Solution', index=False)

            self.log(f"\nExported results to: {excel_path}")
            messagebox.showinfo("Success", f"Results exported to:\n{excel_path}")

        except Exception as e:
            self.log(f"\nERROR exporting: {e}")
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = ThicknessIterationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
