### Paso 1: Importing Libraries and Packages

El primer paso consiste en instalar e importar todas las librerías y paquetes necesarios para el resto del tutorial. Los paquetes que debemos instalar (es decir, no son incluidos por defecto cuando instalamos Python) son:

- *numpy*
- *pandas*
- *seaborn*
- *matplotlib*
- *scikit-learn*

Podemos instalarlos usando el comando ***pip*** de la siguiente manera:

```bash
pip install <nombre-de-librería>
```

Finalmente, ejecutamos el código incluído en el blog para obtener la siguiente salida:

![Untitled](https://www.datocms-assets.com/106983/1696363088-ut2-pd4-1.png)

### Paso 2: Loading and Viewing Data Set

En este paso cargamos los datos desde el sistema de archivos utilizando funciones de la librería ***pandas***.

Al inspeccionar las primeras filas con la función *head*, encontramos que los datos no se encuentran en un estado óptimo para la construcción de nuestro modelo, por lo que será necesario realizar preprocesamiento de los mismos.

![Untitled](https://www.datocms-assets.com/106983/1696363092-ut2-pd4-2.png)

![Untitled](https://www.datocms-assets.com/106983/1696363097-ut2-pd4-3.png)

### Paso 3: Dealing with NaN Values (Imputation)

Al inspeccionar los datos, podemos encontrar valores faltantes en la columna de la edad (***Age***) y el número de cabina (***cabin***) principalmente. Estos valores faltantes generarán problemas a la hora de construir nuestro modelo, por lo que necesitamos rellenarlos.

![Untitled](https://www.datocms-assets.com/106983/1696363101-ut2-pd4-4.png)

En el proceso de *emprolijar* los datos para generar el modelo, además de rellenar valores faltantes, también eliminaremos atributos complicados que no parecen añadir información a las predicciones que deseamos realizar (si el pasajero sobrevivió o no). Para esto, eliminaremos los campos ***cabin*** (que ya era muy problemático por la cantidad de valores faltantes) y ***ticket.***

![Untitled](https://www.datocms-assets.com/106983/1696363106-ut2-pd4-5.png)

![Untitled](https://www.datocms-assets.com/106983/1696363111-ut2-pd4-6.png)

Ahora solo queda rellenar los valores de edad. Para conocer más sobre estos valores, específicamente su distribución en el rango, aprovechamos la librería ***seaborn***:

![Untitled](https://www.datocms-assets.com/106983/1696363116-ut2-pd4-7.png)

![Figure_1.png](https://www.datocms-assets.com/106983/1696363083-ut2-pd4-figure_1.png)

Ya que en el gráfico podemos observar una leve inclinación hacia la derecha (es decir, los valores más altos del rango), llenaremos los valores vacíos con su mediana. Elegimos la mediana en lugar de la media porque los valores altos que se dan en la inclinación tienden a influenciar más la última, a diferencia de la primera.

![Untitled](https://www.datocms-assets.com/106983/1696363120-ut2-pd4-8.png)

![Untitled](https://www.datocms-assets.com/106983/1696363124-ut2-pd4-9.png)

### Paso 4: Plotting and Visualizing Data

Para tener un mejor entendimiento de los datos, su distribución y sus correlaciones, es importante ser capaces de visualizarlos adecuadamente. Aprovechando las librerías importadas, vamos a generar diferentes ilustraciones de la información disponible.

![Untitled](https://www.datocms-assets.com/106983/1696363128-ut2-pd4-10.png)

![Con este diagrama de barras podemos notar inmediatamente que los pasajeros del género femenino tenían una probabilidad inmensamente mayor de sobrevivir que aquellos del género masculino.](https://www.datocms-assets.com/106983/1696363050-ut2-pd4-figure_1-1.png)

Con este diagrama de barras podemos notar inmediatamente que los pasajeros del género femenino tenían una probabilidad inmensamente mayor de sobrevivir que aquellos del género masculino.

![Untitled](https://www.datocms-assets.com/106983/1696363133-ut2-pd4-11.png)

![Los datos indican que casi el 70% de los pasajeros que sobrevivieron eran mujeres.](https://www.datocms-assets.com/106983/1696363137-ut2-pd4-12.png)

Los datos indican que casi el 70% de los pasajeros que sobrevivieron eran mujeres.

Claramente, el género es un factor clave para la predicción de supervivencia de un pasajero.

A continuación, se explora la influencia del atributo ***class***.

![Untitled](https://www.datocms-assets.com/106983/1696363141-ut2-pd4-13.png)

![Podemos observar que los pasajeros que tenían pasajes de mayor clase tuvieron una mayor tendencia a sobrevivir. Esto podría ser por la ubicación de sus cabinas en el barco o la prioridad a la hora de asignar botes salvavidas.](https://www.datocms-assets.com/106983/1696363055-ut2-pd4-figure_1-2.png)

Podemos observar que los pasajeros que tenían pasajes de mayor clase tuvieron una mayor tendencia a sobrevivir. Esto podría ser por la ubicación de sus cabinas en el barco o la prioridad a la hora de asignar botes salvavidas.

![Untitled](https://www.datocms-assets.com/106983/1696362965-ut2-pd4-14.png)

![Los datos indican que, dentro de su propia clase, los pasajeros de primera y tercera clase tuvieron una mayor tendencia a sobrevivir que aquellos de segunda clase.](https://www.datocms-assets.com/106983/1696362971-ut2-pd4-15.png)

Los datos indican que, dentro de su propia clase, los pasajeros de primera y tercera clase tuvieron una mayor tendencia a sobrevivir que aquellos de segunda clase.

![Untitled](https://www.datocms-assets.com/106983/1696362975-ut2-pd4-16.png)

![Figure_1.png](https://www.datocms-assets.com/106983/1696363060-ut2-pd4-figure_1-3.png)

![Untitled](https://www.datocms-assets.com/106983/1696362980-ut2-pd4-17.png)

![Figure_1.png](https://www.datocms-assets.com/106983/1696363064-ut2-pd4-figure_1-4.png)

Viendo como los datos son afectados por la clase de los pasajeros, también podemos concluir que es un atributo importante a la hora de predecir su supervivencia.

Inspeccionando profundamente el atributo de edad, podemos encontrar hallazgos interesantes:

![Untitled](https://www.datocms-assets.com/106983/1696362984-ut2-pd4-18.png)

![Figure_1.png](https://www.datocms-assets.com/106983/1696363068-ut2-pd4-figure_1-5.png)

![Untitled](https://www.datocms-assets.com/106983/1696362990-ut2-pd4-19.png)

![Figure_1.png](https://www.datocms-assets.com/106983/1696363073-ut2-pd4-figure_1-6.png)

Al observar las tendencias de estos diagramas hacia los valores menores en el rango de edad, podemos concluir que los pasajeros más jovenes (entre 20 y 35 años) son quienes tuvieron la mayor tendencia a sobrevivir.

Finalmente, generamos una compilación de gráficos con las relaciones entre los distintos atributos

![Untitled](https://www.datocms-assets.com/106983/1696362995-ut2-pd4-20.png)

![Figure_1.png](https://www.datocms-assets.com/106983/1696363077-ut2-pd4-figure_1-7.png)

### Paso 5: Feature Engineering

Para poder usar estos datos en nuestro modelo, será necesario convertir los atributos ***sex*** y ***embarked*** a valores númericos. Para esto, ya que el sexo es un atributo binario, lo reemplazaremos con 1s y 0s, donde 1 es para género femenino y 0 para masculino. Por otro lado, los valores del campo ***embarked*** son 3: S, C y Q, por lo que una conversión similar es posible. A los valores mencionados anteriormente se le asignan los siguientes valores númericos respectivamente: 0, 1 y 2.

Para realizar esta conversión de datos, aprovecharemos la librería *scikit-learn:*

![Untitled](https://www.datocms-assets.com/106983/1696362999-ut2-pd4-21.png)

![Podemos observar que los valores de los atributos ***sex*** y ***embarked*** han sido transformados a sus correspondientes númericos como definimos previamente.](https://www.datocms-assets.com/106983/1696363003-ut2-pd4-22.png)

Podemos observar que los valores de los atributos ***sex*** y ***embarked*** han sido transformados a sus correspondientes númericos como definimos previamente.

**Creating Synthetic Features**

Puede ser útil generar atributos que creemos que puedan ser de utilidad para nuestro modelo, tomando como base los datos existentes.

Combinaremos los campos ***sibsip*** y ***parch*** para generar el atributo ***famsize***. Los atributos originales representan la cantidad de hermanos y hermanas y la cantidad de padres e hijos del pasajero, respectivamente. Al combinarlos en un único atributo, tendremos información general sobre la cantidad de familiares dentro del barco para cada pasajero.

![Untitled](https://www.datocms-assets.com/106983/1696363008-ut2-pd4-23.png)

En base a este atributo generado, también podemos derivar otro atributo llamado ***isalone***. Este atributo nos indicará si el pasajero abordó el barco sin compañía, lo cual puede ser relevante para su supervivencia.

![Untitled](https://www.datocms-assets.com/106983/1696363013-ut2-pd4-24.png)

Quizás sea posible también extraer información de los nombres de los pasajeros, principalmente relacionado a él título que el pasajero podría haber tenido (señor, señorita, doctor, etc).

![Untitled](https://www.datocms-assets.com/106983/1696363017-ut2-pd4-25.png)

![Las primeras filas de los datos de entrenamiento y de prueba, con los atributos sintéticos agregados.](https://www.datocms-assets.com/106983/1696363021-ut2-pd4-26.png)

Las primeras filas de los datos de entrenamiento y de prueba, con los atributos sintéticos agregados.

![Untitled](https://www.datocms-assets.com/106983/1696363026-ut2-pd4-27.png)

![Podemos ver una variedad amplia de títulos y la frecuencia con la que se dan en los datos del conjunto de entrenamiento.](https://www.datocms-assets.com/106983/1696363030-ut2-pd4-28.png)

Podemos ver una variedad amplia de títulos y la frecuencia con la que se dan en los datos del conjunto de entrenamiento.

Ya que el campo de título nuevamente se trata de un campo de texto, realizaremos una conversión de éste a un campo númerico, reemplazando los menos influyentes por el valor *“Otro”*. Los valores resultantes y su número correspondiente son: *“Señorita”* (0), *“Señor”* (1), *“Señora”* (2), *“Maestro”* (3), *“Doctor”* (4), *“Reverendo”* (5) y *“Otro”* (6).

![Untitled](https://www.datocms-assets.com/106983/1696363036-ut2-pd4-29.png)

Habiendo extraído esta información, el campo ***name*** se vuelve innecesario, por lo que podemos descartarlo de los conjuntos de datos.

![Untitled](https://www.datocms-assets.com/106983/1696363040-ut2-pd4-30.png)

![Untitled](https://www.datocms-assets.com/106983/1696363045-ut2-pd4-31.png)

Con todos los atributos convertidos a valores numéricos, nuestros datos ya están técnicamente listos para utilizarlos para entrenar y probar nuestro modelo. Sin embargo, para garantizar la calidad del mismo, todavía deberíamos reescalar los valores de los campos de *age* y ***fare***, por su amplio rango.

