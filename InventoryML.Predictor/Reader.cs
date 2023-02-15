using InventoryML.ContextLayer;
using Microsoft.ML;

namespace InventoryML.Predictor
{
    public sealed class Reader
    {
        private MLContext _context;

        public Reader(InventoryContext context)
        {
            _context = context.Context;
        }

        public void GetOutput(string modelPath, string imagePath)
        {
            var model = _context.Model.Load(modelPath, out var inputSchema);
            var imageData = _context.Data.LoadFromTextFile<ImageData>(imagePath, hasHeader: false);

            // Perform image classification
            var predictionEngine = _context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictionEngine.Predict((ImageData)imageData);

            // Output the results
            Console.WriteLine("Predicted Label: " + prediction.PredictedLabel);
            Console.WriteLine("Score: " + prediction.Score.Max());
        }
    }
}
