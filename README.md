# syz_v2 kodu full proje

* model.py dosyasında ''' ''' içinde finetune kısmı var orayı açıp yapın fine tune edicez modeli, scirptte kendisi conv layer bulup fine tune ediyor. 
* bu versiyon colabde çalışmaya uygun değil localde çalışır ve belirlediğiniz pathlere gider 
* pathleri güncellemeyi unutmayın
* ana kod dosyamız main.ipynb, onun dışındaki modüllerin içinde fonksiyonalite var yalnızca
* save_model_info.py module ü model hakkındaki her şeyi kaydediyor belirlediğiniz pathlere

### COLAB de çalıştırmak için; 
1. main.ipynb yi notebook olarak açın
2. Diğer fileları yükkleyin (dataloader yerine ggl_data_loader, save_info yerine ggl_save_info)
3. Driveınıza bağlanın kodun içinde image directory pathini belirtin
4. Bloğu çalıştırınca main.ipynb çalışır duruma gelecek
   
