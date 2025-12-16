--------------------------------------------------------------------------------
-- Testbench for FixFunctionByTable_Freq1_uid2
-- Drives stimulus vectors from a CSV file and logs DUT outputs.
--
-- Usage:
--   1. Create simulation input file (one integer per line) at
--        simulation/FixFunctionByTable_inputs.csv
--      Each line should contain a decimal integer representing X.
--   2. Run the testbench in your simulator (xsim, xelab/xsim, ModelSim, etc.)
--   3. The testbench writes results to
--        simulation/FixFunctionByTable_outputs.csv
--      Each line: "<input> <output>"
--
-- The DUT is purely combinational; a dummy clock is included to match the
-- supervisor workflow and provide deterministic sampling points.
--------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity FixFunctionByTable_tb is
end FixFunctionByTable_tb;

architecture sim of FixFunctionByTable_tb is
    constant CLK_PERIOD : time := 10 ns;
    constant OUTPUT_FILE_PATH : string := "simulation/FixFunctionByTable_outputs.csv";

    signal clk_tb   : std_logic := '0';
    signal x_tb     : std_logic_vector(2 downto 0) := (others => '0');
    signal y_tb     : std_logic_vector(10 downto 0);
    signal sample_valid : std_logic := '0';
begin
    -- Clock generation (useful for aligning with other flows, even though DUT is combinational)
    clk_process : process
    begin
        clk_tb <= '0';
        wait for CLK_PERIOD / 2;
        clk_tb <= '1';
        wait for CLK_PERIOD / 2;
    end process;

    -- DUT instantiation
    dut : entity work.FixFunctionByTable_Freq1_uid2
        port map (
            X => x_tb,
            Y => y_tb
        );

    -- Stimulus and logging
    stimulus_process : process
        file output_file : text open write_mode is OUTPUT_FILE_PATH;
        variable line_out : line;
    begin
        report "Sweeping all 8 input vectors for FixFunctionByTable." severity note;

        wait for 5 * CLK_PERIOD;

        for in_value in 0 to 7 loop
            x_tb <= std_logic_vector(to_unsigned(in_value, x_tb'length));
            sample_valid <= '1';

            wait for CLK_PERIOD;

            -- Log input/output pair
            write(line_out, integer'image(in_value));
            write(line_out, string'(" "));
            write(line_out, integer'image(to_integer(signed(y_tb))));
            writeline(output_file, line_out);

            sample_valid <= '0';
        end loop;

        report "Completed sweep. Results written to " & OUTPUT_FILE_PATH severity note;
        wait for 10 * CLK_PERIOD;
        report "Ending simulation." severity note;
        std.env.stop;
    end process;

end sim;

