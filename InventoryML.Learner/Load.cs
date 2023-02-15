using InventoryML.ContextLayer;
using Microsoft.ML;
using System.Data;

namespace InventoryML.Learner
{
    public sealed class Load
    {
        private MLContext _context;

        /// <summary>
        /// Load images into ML pipeline to persist the algorithm thru different files
        /// We should then be able to test the data to predict the outcome of what the file text is
        /// </summary>
        public Load(InventoryContext inventory)
        {
            _context = inventory.Context;
        }

        public void Content()
        {

            var schemas = Directory.GetFiles("images").AsEnumerable();

            // Load the training data
            foreach (var schema in schemas)
            {
                var data = _context.Data.LoadFromTextFile<InputData>(schema, 
                hasHeader: true,
                allowQuoting: true);

                // Split the data into training and test sets
                var trainData = _context.Data.TrainTestSplit(data, testFraction: 0.2);

                // Define the training pipeline
                var pipeline = _context.Transforms.Conversion.MapValueToKey("Label")
                    .Append(_context.Transforms.Concatenate("Features", "Feature1", "Feature2"))
                    .Append(_context.BinaryClassification.Trainers.FieldAwareFactorizationMachine(
                        labelColumnName: "Label",
                        featureColumnName: "Features"));

                // Train the model
                var model = pipeline.Fit(trainData.TrainSet);

                // Persist the model to disk
                _context.Model.Save(model, trainData.TrainSet.Schema, "model.zip");
                
            }

        }

    }
}
