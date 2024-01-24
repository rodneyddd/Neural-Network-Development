using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

public class DataLoader
{
    public static List<DataPoint> LoadData(string filePath)
    {
        var dataPoints = new List<DataPoint>();
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = ",",
            HasHeaderRecord = true,
        };

        using (var reader = new StreamReader(filePath))
        using (var csv = new CsvReader(reader, config))
        {
            while (csv.Read())
            {
                var dataPoint = csv.GetRecord<DataPoint>();
                dataPoints.Add(dataPoint);
            }
        }

        return dataPoints;
    }

    public static List<DataPoint> PreprocessData(List<DataPoint> dataPoints)
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
