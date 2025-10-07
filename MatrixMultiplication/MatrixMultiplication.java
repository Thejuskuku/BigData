import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MatrixMultiplication {

    // Mapper Class 
    public static class MatrixMapper extends Mapper<LongWritable, Text, Text, Text> {

        private int M_ROWS; // Rows of A (C)
        private int N_COLS; // Cols of B (C)
        
        // Read the dimensions from the Configuration in the setup phase
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            M_ROWS = conf.getInt("M_ROWS", 2); // Default to 2 if not set
            N_COLS = conf.getInt("N_COLS", 2); // Default to 2 if not set
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Input: [Matrix ID], [Row/Col Index], [Col/Row Index], [Value]
            String[] parts = value.toString().split(",");
            String matrix = parts[0]; // A or B
            int row = Integer.parseInt(parts[1].trim());
            int col = Integer.parseInt(parts[2].trim());
            String val = parts[3].trim();
            
            // The multiplication C(i, j) = sum_k A(i, k) * B(k, j)
            
            if (matrix.equals("A")) {
                // A(i, k) contributes to C(i, j) for all j = 0 to N-1
                // i is the final row, k is the inner index
                for (int j = 0; j < N_COLS; j++) {
                    // Key: "i,j" (Final C position)
                    String outputKey = row + "," + j;
                    // Value: "A,k,value" (Matrix ID, inner index, value)
                    String outputValue = "A," + col + "," + val; 
                    context.write(new Text(outputKey), new Text(outputValue));
                }
            } else if (matrix.equals("B")) {
                // B(k, j) contributes to C(i, j) for all i = 0 to M-1
                // j is the final col, k is the inner index
                for (int i = 0; i < M_ROWS; i++) {
                    // Key: "i,j" (Final C position)
                    String outputKey = i + "," + col;
                    // Value: "B,k,value" (Matrix ID, inner index, value)
                    String outputValue = "B," + row + "," + val; 
                    context.write(new Text(outputKey), new Text(outputValue));
                }
            }
        }
    }

    // Reducer Class 
    public static class MatrixReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Stores A values: Map<k, value_A_ik>
            Map<Integer, Float> matrixA = new HashMap<>();
            // Stores B values: Map<k, value_B_kj>
            Map<Integer, Float> matrixB = new HashMap<>();
            
            // 1. Group the values by matrix tag and index k
            for (Text val : values) {
                // val is "A,k,value" or "B,k,value"
                String[] parts = val.toString().split(",");
                String matrix = parts[0];
                int k = Integer.parseInt(parts[1].trim());
                float value = Float.parseFloat(parts[2].trim());

                if (matrix.equals("A")) {
                    matrixA.put(k, value);
                } else {
                    matrixB.put(k, value);
                }
            }
            
            // 2. Perform the multiplication and summation
            float result = 0.0f;
            
            // Iterate over the common inner index k
            for (Integer k : matrixA.keySet()) {
                if (matrixB.containsKey(k)) {
                    // C(i, j) += A(i, k) * B(k, j)
                    float valueA = matrixA.get(k);
                    float valueB = matrixB.get(k);
                    result += valueA * valueB;
                }
            }

            // Output: "i,j" \t "C_ij_value"
            context.write(key, new Text(String.valueOf(result)));
        }
    }

    // Driver (Main) Function
    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: MatrixMultiplication <A_rows M> <B_cols N> <input path> <output path> <A_cols/B_rows K>");
            System.exit(-1);
        }
        
        // Define dimensions based on command line arguments
        int M = Integer.parseInt(args[0]); // A_rows
        int N = Integer.parseInt(args[1]); // B_cols
        int K = Integer.parseInt(args[4]); // A_cols / B_rows

        Configuration conf = new Configuration();
        
        // Pass M and N to the Mapper using Configuration
        conf.setInt("M_ROWS", M);
        conf.setInt("N_COLS", N);
        conf.setInt("K_COLS", K); // K is not strictly needed but good practice

        Job job = Job.getInstance(conf, "Matrix Multiplication");

        job.setJarByClass(MatrixMultiplication.class);
        job.setMapperClass(MatrixMapper.class);
        job.setReducerClass(MatrixReducer.class);

        // Set output types for the job (matches the Reducer output)
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Input/Output paths are passed via command-line args
        FileInputFormat.addInputPath(job, new Path(args[2])); // input path
        FileOutputFormat.setOutputPath(job, new Path(args[3])); // output path

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}