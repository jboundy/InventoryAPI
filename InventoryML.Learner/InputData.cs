using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InventoryML.Learner
{
    public sealed class InputData
    {
        [LoadColumnAttribute(1)]
        public float Feature1 { get; set; }
        [LoadColumnAttribute(2)]
        public float Feature2 { get; set; }
        [LoadColumnAttribute(3)]
        public bool Label { get; set; }
    }
}
