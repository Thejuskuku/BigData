import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

public class MaxTemperature {

    // Mapper class 
    public static class MaxTemperatureMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

        private Text year = new Text();
        private IntWritable temp = new IntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            // Assuming the year is the first 4 characters and temp starts at a fixed position
            // In a real-world scenario, parsing would be more robust (e.g., split by space/tab)
            String yearStr = line.substring(0, 4);
            int temperature = Integer.parseInt(line.substring(5).trim());

            year.set(yearStr);
            temp.set(temperature);

            // Emit the (Year, Temperature) pair
            context.write(year, temp);
        }
    }

    // Reducer class 
    public static class MaxTemperatureReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int maxValue = Integer.MIN_VALUE;

            // Iterate through all temperatures for the given year (key)
            for (IntWritable val : values) {
                if (val.get() > maxValue) {
                    maxValue = val.get();
                }
            }

            result.set(maxValue);
            // Emit the (Year, MaxTemperature) pair
            context.write(key, result);
        }
    }

    // Main function to set up and run the job 
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: MaxTemperature <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Max Temperature");

        job.setJarByClass(MaxTemperature.class);
        job.setMapperClass(MaxTemperatureMapper.class);
        // The combiner can be the same as the reducer to find max temps on the map side first
        job.setCombinerClass(MaxTemperatureReducer.class);
        job.setReducerClass(MaxTemperatureReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
