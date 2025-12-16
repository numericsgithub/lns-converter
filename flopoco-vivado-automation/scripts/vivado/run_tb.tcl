###############################################################################
# TCL script to compile and run FixFunctionByTable testbench with xsim.
# Usage:
#   vivado -mode batch -source run_tb.tcl
# or
#   xsim -t run_tb.tcl
###############################################################################

# Clean existing design sources
if {[file exists xsim.dir]} {
    file delete -force xsim.dir
}

read_vhdl hdl/flopoco.vhdl
read_vhdl tb_FixFunctionByTable.vhdl

set_property top FixFunctionByTable_tb [current_fileset]

launch_simulation
run all
puts "Simulation is done"
# Ensure simulator shuts down cleanly
quit

