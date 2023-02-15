using Microsoft.ML;

namespace InventoryML.ContextLayer
{
    public sealed class InventoryContext
    {
        public MLContext Context;
        public InventoryContext(MLContext context)
        {
            Context = context;
        }
    }
}
