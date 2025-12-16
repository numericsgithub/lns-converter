#!/usr/bin/env tclsh
#
# Project-free Vivado batch flow.
# Usage (from Python helper):
#   vivado -mode batch -source project_free_flow.tcl -tclargs <part> <top> <sources_dir> <output_dir> <constraint1> ...
#

proc print_usage {} {
    puts "Usage: project_free_flow.tcl <part_name> <top_module> <sources_dir> <output_dir> ?constraint_files...?"
    puts "Example:"
    puts "    vivado -mode batch -source project_free_flow.tcl -tclargs xczu3eg-sbva484-1-e top ./hdl ./build ./constraints/top.xdc"
}

if {[llength $argv] < 4} {
    puts "ERROR: Not enough arguments."
    print_usage
    exit 1
}

set part_name    [lindex $argv 0]
set top_name     [lindex $argv 1]
set sources_dir  [file normalize [lindex $argv 2]]
set output_dir   [file normalize [lindex $argv 3]]
set constraint_files [lrange $argv 4 end]

if {![file isdirectory $sources_dir]} {
    puts "ERROR: Sources directory not found: $sources_dir"
    exit 1
}

file mkdir $output_dir
set log_dir [file join $output_dir logs]
file mkdir $log_dir

set source_files {}
foreach pattern {*.v *.sv *.vh *.vhd *.vhdl} {
    set matches [glob -nocomplain -directory $sources_dir $pattern]
    if {[llength $matches] > 0} {
        set source_files [concat $source_files $matches]
    }
}

# Filter out testbench files (they should not be synthesized)
set filtered_files {}
foreach file $source_files {
    set filename [file tail $file]
    # Skip files that start with 'tb_' or end with '_tb' (testbench naming convention)
    if {![string match "tb_*" $filename] && ![string match "*_tb.*" $filename] && ![string match "*_tb_*" $filename]} {
        lappend filtered_files $file
    } else {
        puts "INFO: Skipping testbench file: $filename"
    }
}
set source_files $filtered_files

if {[llength $source_files] == 0} {
    puts "ERROR: No source files found in $sources_dir"
    exit 1
}

puts "Reading HDL sources..."
foreach file $source_files {
    switch -nocase -- [file extension $file] {
        ".v" -
        ".sv" -
        ".vh" { read_verilog -sv $file }
        ".vhd" -
        ".vhdl" { read_vhdl $file }
        default { puts "WARNING: Skipping unsupported file $file" }
    }
}

puts "Setting part to $part_name"
set_part $part_name

if {[llength $constraint_files] > 0} {
    puts "Reading constraint files..."
    foreach xdc $constraint_files {
        set normalized [file normalize $xdc]
        if {[file exists $normalized]} {
            read_xdc $normalized
        } else {
            puts "WARNING: Constraint file not found: $normalized"
        }
    }
}

set synth_checkpoint [file join $output_dir "post_synth.dcp"]
set impl_checkpoint  [file join $output_dir "post_route.dcp"]
set bitstream_file   [file join $output_dir "$top_name.bit"]
set reports_dir      [file join $output_dir reports]
file mkdir $reports_dir

puts "Starting synthesis..."
synth_design -top $top_name -part $part_name -flatten_hierarchy rebuilt
write_checkpoint -force $synth_checkpoint
report_utilization -file [file join $reports_dir "util_post_synth.rpt"]

puts "Running opt/place/route..."
opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force $impl_checkpoint
report_timing_summary -file [file join $reports_dir "timing_summary.rpt"]
report_power -file [file join $reports_dir "power.rpt"]
report_utilization -file [file join $reports_dir "util_post_route.rpt"] -hierarchical -hierarchical_depth 2

puts "Project-free flow completed successfully."
exit 0

