using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

public class DataLoader
{
    public static List<DataPoint> LoadData(string filePath) //the function returns a list of datapoints
    //the parameter is the path to the CSV file that contains the data.
    {
        var dataPoints = new List<DataPoint>(); //this is a variable that holds a new empty list of DataPoint objects to hold the data read from the CSV file.
        
        /*
        This section creates a configuration object (config) for reading CSV files using the CsvHelper library. 
        It specifies that the CSV files use a comma (,) as the delimiter between fields and that the file has a header record.
        */
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = ",",
            HasHeaderRecord = true,
        };

        /*
        This block opens the CSV file located at filePath using a StreamReader and 
        creates a CsvReader object (csv) with the specified configuration (config). 
        The using statement ensures that the StreamReader and CsvReader objects are properly disposed of after use.
        */


        using (var reader = new StreamReader(filePath))
        using (var csv = new CsvReader(reader, config))
        {
            /*
            This while loop reads each line of the CSV file using the Read method of the CsvReader object (csv). 
            Inside the loop, it uses the GetRecord<DataPoint>() method to deserialize the current line 
            of the CSV file into a DataPoint object and adds it to the dataPoints list.
            */

            /*
            The CsvReader object csv has a method called Read().
            Each time csv.Read() is called, it attempts to read the next line of the CSV file.
            If there is a line to read, csv.Read() returns true, and the loop continues executing.
            If there are no more lines to read (the end of the file is reached), csv.Read() returns false, and the loop stops executing.
            Therefore, the while loop continues iterating as long as there are more lines to read in the CSV file. 
            It stops when there are no more lines to read, effectively processing the entire contents of the CSV file.
            */
            while (csv.Read())
            {
                var dataPoint = csv.GetRecord<DataPoint>();
                dataPoints.Add(dataPoint);
            }
        }

        return dataPoints;
    }

    public static List<DataPoint> PreprocessData(List<DataPoint> dataPoints)
    //takes a list of DataPoint objects as input and returns a list of DataPoint objects after preprocessing.
    {
        // Example normalization (adjust according to your dataset)
        double maxTemp = double.MinValue;
        foreach (var dataPoint in dataPoints)
        {
            if (dataPoint.AbsoluteTemperature > maxTemp)
                maxTemp = dataPoint.AbsoluteTemperature;
        }

        foreach (var dataPoint in dataPoints) 
        {
            dataPoint.AbsoluteTemperature /= maxTemp; // Simple normalization
        }

        return dataPoints;
    }
    /*
    This function preprocesses a list of data points by normalizing the 
    AbsoluteTemperature attribute to be within the range [0, 1] 
    based on the maximum temperature value found in the dataset. 
    It's a common preprocessing step in machine learning and data analysis tasks to ensure that the features are on similar scales,
    which can help improve the performance and convergence of models.
    */
}

public class DataPoint
{
    public double AbsoluteTemperature { get; set; }
    public double RelativeLuminosity { get; set; }
    public double RelativeRadius { get; set; }
    public double AbsoluteMagnitude { get; set; }
    public string StarColor { get; set; }
    public string SpectralClass { get; set; }
    public int StarType { get; set; }
}
//as for this clas, here we have datapoints for each variable 
/*
Each instance of the DataPoint class holds values for these attributes, 
providing a structured representation of the data associated with each star in the dataset. 
The DataPoint class allows for convenient organization and manipulation of star data, 
enabling various analysis and processing tasks to be performed on the dataset.

basically this is where you store variables relating to the csv
*/