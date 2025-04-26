Code in this folder is derived from Puffin's GitHub repository (commit: [34a7278](https://github.com/jzhoulab/puffin/commit/34a7278954382ca8ca1bf15ec19d41191bf215ee)).

> For training Puffin-D, you need to install the `custom_target_support` branch of Selene
> 
> ```
> git clone https://github.com/kathyxchen/selene.git
> cd selene
> git checkout custom_target_support
> python setup.py build_ext --inplace
> python setup.py install 
> ```

Other dependent packages:
* pyfaidx
* pytorch

Modifications compared to the original implementation:
* No blacklist for FANTOM CAGE was included since we didn't train on CAGE
* 