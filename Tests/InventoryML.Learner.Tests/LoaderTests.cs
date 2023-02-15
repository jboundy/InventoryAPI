using InventoryML.Learner;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InventoryML.Learner.Tests
{
    [TestClass]
    public class LoaderTests
    {
        private MLContext _context;

        public LoaderTests()
        {
            _context = new MLContext();
        }

        [TestMethod()]
        public void CanSaveModelToDisk()
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

            var loadedModel = _context.Model.Load("model.zip", out var inputSchema);

            Assert.IsTrue(loadedModel != null);
        }
    }
}
