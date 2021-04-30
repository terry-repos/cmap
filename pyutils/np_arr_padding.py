import syssys.path.append('..')import numpy as npdef pad_2d_lobsided(arrToPad, padVal, position=None, fillValue=np.nan): #position need be one of top bottom left right    r, c = arrToPad.shape    if ((position=='T') or (position=='B')):        r+=padVal    if ((position=='L') or (position=='R')):        c+=padVal    outArr = np.full(shape=(r,c), fill_value=fillValue, dtype=arrToPad.dtype)    if position=='T':        outArr[padVal : r, 0 : c] = arrToPad    if position=='B':        outArr[0 : (r-padVal), 0 : c] = arrToPad    if position=='L':        outArr[0 : r, padVal : c] = arrToPad    if position=='R':        outArr[:, 0 : (c-padVal)] = arrToPad    return outArrdef pad_rgb_2d_lobsided(arrToPad, padVal, position=None): #position need be one of top bottom left right    r, c, rgb = arrToPad.shape    if ((position=='T') or (position=='B')):        r+=padVal    if ((position=='L') or (position=='R')):        c+=padVal    outArr = np.full(shape=(r,c,rgb), fill_value=np.nan, dtype=arrToPad.dtype)    if position=='T':        outArr[padVal : r, 0 : c, :] = arrToPad    if position=='B':        outArr[0 : (r-padVal), 0 : c, :] = arrToPad    if position=='L':        outArr[0 : r, padVal : c, :] = arrToPad    if position=='R':        outArr[0 : r, 0 : (c-padVal), :] = arrToPad    return outArrdef pad_2d_edges(arrToPad, padVal):    m,n = arrToPad.shape    paddedArr = np.full(shape=(m+padVal*2, n+padVal*2), fill_value=np.nan, dtype=arrToPad.dtype)    paddedArr[padVal : (m+padVal), padVal : (n+padVal)] = arrToPad    return paddedArrdef unpad_2d_edges(arrToUnPad, padVal):    m,n = arrToUnPad.shape    unPaddedArr = arrToUnPad[padVal : (m+padVal), padVal : (n+padVal)]                           return unPaddedArrdef unpad_2d_edges_after_resX(arrToUnPad, addToEndPadL=0, overWriteResX=0):    global padL    global resXi    if overWriteResX == 0:        overWriteResX = resXi    begZ = (padL+1+addToEndPadL) * ((2)**(overWriteResX-2))  #Left and top is where nans begin. should this be padL * 2?    endZ = (padL+addToEndPadL) * ((2)**(overWriteResX-2))+1#  The bigger EndZ, the smaller the array         unPaddedArr = arrToUnPad[(begZ):(arrToUnPad.shape[0] - endZ), (begZ) : (arrToUnPad.shape[1] - endZ)]                           return unPaddedArrdef pad_3d_edges(anInArr, padding=None):    global animationPaddingMultiplier    if padding==None:        pv = (int)(math.ceil(anInArr.shape[1] * animationPaddingMultiplier))    else:        pv = padding    h = anInArr.shape[1]    w = anInArr.shape[2]    newH = h + 2 * pv    newW =  w + 2 * pv    paddedWaveArr = np.full(shape=(anInArr.shape[0], newH, newW, 4), fill_value=1.0, dtype="float32")    paddedWaveArr[:, pv : (pv + h), pv : (pv + w), :] = anInArr    return paddedWaveArr, pvdef pad_rgb_2d_edges(arrToPad, padVal):    m, n, rgb = arrToPad.shape    paddedArr = np.full(shape=(m+padVal*2, n+padVal*2, rgb), fill_value=np.nan, dtype=arrToPad.dtype)    paddedArr[padVal : (m+padVal), padVal : (n+padVal), :] = arrToPad    return paddedArr    def insert_alt_rows(arrToAddRows, val=np.NaN):    newNrows = arrToAddRows.shape[0]*2    arrWithNewRows = np.full(shape=(newNrows, arrToAddRows.shape[1]), fill_value=val, dtype=arrToAddRows.dtype)    arrWithNewRows[1::2,] = arrToAddRows    # print(str(arrToAddRows.shape) + " _ new rows shape _ " + str(arrWithNewRows.shape)    return arrWithNewRowsdef insert_alt_cols(arrToAddCols, val=np.NaN):    newNcols = arrToAddCols.shape[1]*2    arrWithNewCols = np.full(shape=(arrToAddCols.shape[0], newNcols), fill_value=val, dtype=arrToAddCols.dtype)    arrWithNewCols[:,1::2] = arrToAddCols    # print(str(arrToAddCols.shape) + " _ new cols shape _ " + str(arrWithNewCols.shape)    return arrWithNewColsdef insert_alt_rows_and_cols(arrToAddRowsAndCols, val=np.nan): #Handles 2d and 3d    if (len(arrToAddRowsAndCols.shape)==2): #if 2D        altRowsArr = insert_alt_rows(np.copy(arrToAddRowsAndCols),val)        altColsAndRowsArr = insert_alt_cols(altRowsArr,val)    else: #assume 3D        newNrows = arrToAddRowsAndCols.shape[0]*2        newNcols = arrToAddRowsAndCols.shape[1]*2        altColsAndRowsArr3D = np.full(shape=(newNrows,newNcols,arrToAddRowsAndCols.shape[2]), fill_value=np.nan, dtype=arrToAddRowsAndCols.dtype)        for x in range(arrToAddRowsAndCols.shape[2]):            altRowsArr = insert_alt_rows(arrToAddRowsAndCols[:,:,x],val)            altColsAndRowsArr3D[:,:,x] = insert_alt_cols(altRowsArr,val)          altColsAndRowsArr = altColsAndRowsArr3D    return altColsAndRowsArr# Make a simple copy of the outermost elementsdef extend_edges(inArr):    h = inArr.shape[0] - 1    w = inArr.shape[1] - 1    newArr = np.copy(inArr)    for r in range(0, h):        for c in range(0, w):            if ~(np.isnan(newArr[r,c])):                newArr[r, c-1]=newArr[r,c]                break                for c in range(w, 0, -1):            if ~(np.isnan(newArr[r,c])):                newArr[r, c+1]=newArr[r,c]                break    for c in range(0, w):        for r in range(0, h):            if ~(np.isnan(newArr[r,c])):                newArr[r-1, c]=newArr[r,c]                break           for r in range(h, 0, -1):            if ~(np.isnan(newArr[r,c])):                newArr[r+1, c]=newArr[r,c]                   break       return newArr   