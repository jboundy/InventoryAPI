#Introduction

##Purpose
To change how inventory is managed

#Steps for process
1) write ML model to read in a receipt. This is what is logged to internal inventory. Data on the receipt is either read if it understands what the product is, or the data is looked up by a back-end service to identify the product.

2)When inventory is taken out, a picture image will be taken. That picture is interpreted by the ML model, identified and take out of inventory from persisted data storage.