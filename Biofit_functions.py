import pandas as pd
import numpy as np
import copy
from scipy.optimize import curve_fit

#Function to define R^2 value - to give quantitative value as to the degree of fit
def Rsqrd(xdata,ydata,func,pop):
    residuals = ydata - func(np.asarray(xdata), *pop)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

##Create file if does not exist
def checkdir(dir):
    #Import functions
    import os
    #First check if directory exists
    if os.path.isdir(dir) == False:
        os.makedirs(dir)
    else:
        pass

#Function to determine average of data sets fed with dataframe of with columns of common xpoints
def avg_set(xdata_sets,ydata_sets,x0_replace):
    #Itterate over datasets considered to get list of unique x_axis data sets
    compiled_xdata = []
    for data in xdata_sets:
        #itterate over each data point in x data sets and add to list if not already in list
        for d in data:
            #if d == 0:
                #d = 1e-13
            #    compiled_xdata.append(1e-8)
            if d in compiled_xdata:
                pass
            else:
                compiled_xdata.append(d)
    #itterate over compiled xdata and replace any 0s with very small number instead
    compiled_xdata = [ x if x!=0 else x0_replace for x in compiled_xdata]
    #print(compiled_xdata)

    #Having produced a compiled xdata set want to make dataframe with columns associated with each x data point
    all_data = pd.DataFrame(columns=compiled_xdata)

    #To find any average of any duplicated x axis data points need to add in each row associated with each data set
    for i,data in enumerate(xdata_sets):
        #replace 0s in x data set with x0_replace
        data = [ x if x!=0 else x0_replace for x in data]
        #make dictionary of add ydata and associated xdata points to add into dataframe
        added_data = {}
        #Itterating over each data point in each data set append to dictionary
        for j,d in enumerate(compiled_xdata):
            #Check if x value in dictionary keys
            if d in data:
                #if x data point in x data set find which integer it corresponds to
                for p,g in enumerate(data):
                    #itterate through list of data and stop at first matching value
                    if g == d:
                        #add corresponding y data point to dictionary from integer found
                        added_data.setdefault(d,ydata_sets[i][p])
            else:
                pass
        #Having made dictionary append row to dataframe
        all_data = all_data.append(added_data, ignore_index=True)
    #Having made dataframe with each row assocaited with each data point want to find average of each y value associated with each x point
    yaverage_set = []
    for col in all_data:
        yaverage_set.append(np.nanmean(all_data[col].to_numpy()))
    return (compiled_xdata,yaverage_set)

#Define function to fit curves to data
def fitting(basal,function,model,xdata,ydata,error,sigma,bounds,abounds):
    if model == 'Inverse_Monod':
        pop, pcov = curve_fit(f=function, xdata=xdata, ydata=ydata, sigma=sigma)
    else:
        #make temp holder of bounds so that can be used in each fitting
        tbounds = copy.deepcopy(bounds)
        #Need to add in any additional bounds on top of those for menten
        for i,a in enumerate(abounds):
            tbounds[i].extend(a)
        #Having defined bounds check which functions to fit and then fit
        if basal == 'Yes':
            #If have basal produciton and using bounds also need to add a term to bounds used
            amax = max(ydata)
            amin = -np.inf
            a_terms = ([amin],[amax])
            for i,a in enumerate(a_terms):
                tbounds[i].extend(a_terms[i])
            #print(tbounds)
            if error == 'Yes':
                pop, pcov = curve_fit(f=function, xdata=xdata, ydata=ydata, sigma=sigma,maxfev=1000000,bounds=tbounds)
            else:
                pop, pcov = curve_fit(f=function, xdata=xdata, ydata=ydata,maxfev=1000000,bounds=tbounds)
        else:
            if error == 'Yes':
                pop, pcov = curve_fit(f=function, xdata=xdata, ydata=ydata, sigma=sigma, maxfev=1000000,bounds=tbounds)
            else:
                pop, pcov = curve_fit(f=function, xdata=xdata, ydata=ydata,maxfev=1000000,bounds=tbounds)
    return (pop, pcov)

#Define function to save values of associated fitted values to specified dataframe
def fit_figures(basal,xdata,ydata,xdata_plot,var_list,pop,model_name,model,df):
    #Having fit function make dictionary of tuples which can then be used to input data into dataframe of all variables
    var_pairs = {var_list[i]:pop[i] for i,j in enumerate(var_list)}
    #Calculate R^2 value for function
    r_squared = Rsqrd(xdata,ydata,model,pop)
    #Calculate the maximum y-axis value calculated by fitted function
    max_function_y = max(model(np.asarray(xdata_plot), *pop))
    #add dictionary entry of the calculated R^2 value
    var_pairs.setdefault('R Squared',r_squared)
    #add dictionary entry to specify the model used
    var_pairs.setdefault('model',model_name)
    #add dictionary entry to detail the maximum y-axis value calcualted from function
    var_pairs.setdefault('max_function_y',max_function_y)
    #Convert dictionary into dataframe and return
    return (df.append(pd.DataFrame(var_pairs,columns=list(var_pairs.keys()),index=[1])),r_squared,max_function_y)


#Function to scale multiple data sets to be within the same range as the dataset with the greatest v_max to enable evaluation of multiple data sets together while
#excluding the effects that different maximum production or proliforation rates may haves - uses linear interpolation
#Rather than just looking for a maximum value and defining a scaling value want to find which data set has the highest production overall and then use that set to
#scale the other data sets defining a scaling factor by the two points closest to a given value in another data set
def data_scalar(ixdata_sets,iydata_sets,iyerr_sets):
    #Initially want to determine which data set has the highest production to do this find the mean average of each production rate
    #make place holder for highest mean average of data sets to identify which has the highest production rate
    set_mean = (0,0)
    #Start by itterating through each data set and calculate the mean average
    for i,s in enumerate(iydata_sets):
        #for each dataset calculate mean production rate and update index value if higher than current mean
        if np.mean(np.asarray(s)) > set_mean[0]:
            set_mean = (np.mean(np.asarray(s)),i)
    #Make place holders for scaled data sets
    sxdata_sets = []
    sydata_sets = []
    syerr_sets = []
    #Having identified the dataset with the highest mean value itterate through other data sets and scale according to linear interpolation
    for i,s in enumerate(iydata_sets):
        #print(s)
        #If data set index is the same as that with the highest value do not need to scale so just append to list of scaled data sets and pass
        if i == set_mean[1]:
            sxdata_sets.append(ixdata_sets[i])
            sydata_sets.append(iydata_sets[i])
            syerr_sets.append(iyerr_sets[i])
        #Otherwise go through and scale values
        else:
            #To scale y data points with interpolation find values on eitherside of each in both x and y axis.
            #make a list of all scale factors
            s_factors = []
            #create place holder to scaled data sets
            sydata_set = []
            syerr_set = []
            #Itterating through data points in data set need to make sure that data sets are comparable
            for j,k in enumerate(ixdata_sets[i]):
                #make place holder for scale factor
                scale_factor = 1
                #check if x-axis value in data set with greatest mean value
                if k not in ixdata_sets[set_mean[1]]:
                    #if value is not in data set with greatest mean value need to interpolate to find comparable y-axis value to determine scaling factor from
                    #check that x-axis value is not larger than the largest value in data set with highest mean value
                    if k < max(ixdata_sets[set_mean[1]]):
                        #Itterate through data set with highest mean value and find values on either side
                        for r,t in enumerate(ixdata_sets[set_mean[1]]):
                            if t > k:
                                x1 = ixdata_sets[set_mean[1]][r-1]
                                x2 = ixdata_sets[set_mean[1]][r]
                                y1 = iydata_sets[set_mean[1]][r-1]
                                y2 = iydata_sets[set_mean[1]][r]
                                #Having identified values on either side interpolate and determine scale factor
                                scale_factor = (y1+((k-x1)*((y2-y1)/(x2-x1))))/iydata_sets[i][j]
                                #print('i scale'+str(scale_factor))
                                #append scale factor to list of scale factors
                                s_factors.append((scale_factor,k))
                                break
                    else:
                       #If the x point is outside that of the largest data set x axis range scale by the difference in maximum dataset mean average y value and the
                        #mean average of the dataset considered
                        scale_factor = set_mean[0]/np.mean(np.asarray(iydata_sets[i]))
                        #print('over scale'+str(scale_factor))
                        #having determined new scale factor then append to list of scale factors
                        s_factors.append((scale_factor,k))
                #If do not need to interpolate to find value go directly ahead and calculate scale factor
                else:
                    scale_factor = iydata_sets[set_mean[1]][j]/iydata_sets[i][j]
                    #print('scale'+str(scale_factor))
                    #append scale factor to list of scale factors
                    s_factors.append((scale_factor,k))
                #having determined scale factor then want to scale value and append to scaled y axis list
                sydata_set.append(iydata_sets[i][j]*scale_factor)
                #Still need to scale y_err set
                #initially look up the percentage error associated with error in original data sets
                syerr_set.append((iyerr_sets[i][j]/iydata_sets[i][j])*(iydata_sets[i][j]*scale_factor))
            #having determined scale list then want to append list to lists of scaled data
            sxdata_sets.append(ixdata_sets[i])
            sydata_sets.append(sydata_set)
            syerr_sets.append(syerr_set)

    #Having scaled all datasets to use then return them
    return (sxdata_sets,sydata_sets,syerr_sets)

#Function to estimate variables of menten and extended models
def comb_set(no_datasets,scale,xdata_sets,ydata_sets,yerr_sets,x0_replace,error):
    #Determine x and y axis data sets from individual or combined datasets
    #Initially consider if need to scale data
    if no_datasets != 1:
        if scale == 'Yes':
            #Scaling data to account for variation in y axis due to intercell variablilty in maximum production or growth rates
            sxdata,sydata,syerr = data_scalar(xdata_sets,ydata_sets,yerr_sets)
            #Combine and average scaled data sets
            xdata,ydata = avg_set(sxdata,sydata,x0_replace)
            if error == 'Yes':
                yerr = avg_set(sxdata,syerr,x0_replace)[1]
                #print(yerr)
            else:
                yerr = []
                pass
        else:
            xdata,ydata = avg_set(xdata_sets,ydata_sets,x0_replace)
            if error == 'Yes':
                yerr = avg_set(xdata_sets,yerr_sets,x0_replace)[1]
                #print(yerr)
            else:
                yerr = []
                pass
    else:
        xdata,ydata = avg_set(xdata_sets,ydata_sets,x0_replace)
        if error == 'Yes':
            yerr = avg_set(xdata_sets,yerr_sets,x0_replace)[1]
            #print(yerr)
        else:
            yerr = []
            pass
    #print(xdata)
    #print(ydata)
    return (xdata,ydata,yerr)

#Function to determine number of steps between x points to plot, want to find average difference between x axis points
#and then take number of steps equal to x_plotno between each x-axis point
def xsteps(xdata,x_plotno,xmin_plot,max_check,max_x):
    #Make list of xaxis differences
    xdif_lst = []
    for i in range(len(xdata)):
        #Want to stop look when difference between last two values has been found
        if i == len(xdata)-1:
            break
        else:
            #calculate difference between x points then append to list
            xdif_lst.append(abs(xdata[i+1]-xdata[i]))
    #convert list to numpy array and then calculate mean average before finding x_plotno of this difference
    xdif_avg = np.mean(np.array(xdif_lst))/x_plotno
    #Make xdif_avg is appropriate to capture smaller values
    if xdif_avg > xdata[1]:
        xdif_avg = xdata[1]
    else:
        pass
    #check if want to plot to maximum x_axis value or higher
    if max_check == 'Yes':
        xdata_plot =  pd.Series(np.arange(xmin_plot,max(xdata),xdif_avg))
    else:
        xdata_plot =  pd.Series(np.arange(xmin_plot,max_x,xdif_avg))

    return xdata_plot

#Function to estimate menten emperical kenetic parameters
def esti_var(Estimated_var,ydata,xdata):
    #For Han and Luong need to to know Smin - this must be a value greater than the largest experimental x-axis value
    Smin = max(xdata)
    if Estimated_var == 'Yes':
        #Estimating variables used in fitting data to curve
        #Take mu or equivilant vmax as the maximum y axis data point
        mu = max(ydata)
        #As the real value to mu may be greater or smaller than the maximum experimental value set mu/vmax estimated bounds to be 10% either side of experimental value
        mu_min = mu - (0.1*mu)
        mu_max = mu + (0.1*mu)
        #Ks is half the concentration at which maximum rate occours to find KS initially find half of maximum rate
        #then determine list indices which either side of half maximum rate to retrieve from x data set
        for i,j in enumerate(ydata):
            if j > max(ydata)/2:
                if i == 0:
                    Ks_max = xdata[i+1]
                    Ks_min = xdata[i+1]*1e-13
                else:
                    Ks_max = xdata[i]
                    Ks_min = xdata[i-1]
                break
        if Ks_min == 0:
            Ks_min = 1e-15
        bounds = {'Menten':([mu_min,Ks_min],[mu_max,Ks_max]),'Inverse_Monod':([mu_min,Ks_min],[mu_max,Ks_max]),'exp_growth':([mu_min],[mu_max]),
        'Han':([mu_min,Ks_min],[mu_max,Ks_max]),'Luong':([mu_min,Ks_min],[mu_max,Ks_max]),'Andrews':([mu_min,Ks_min],[mu_max,Ks_max]),'Aiba':([mu_min,Ks_min],[mu_max,Ks_max]),
        'Moser':([mu_min,Ks_min],[mu_max,Ks_max]),'Edward':([mu_min,Ks_min],[mu_max,Ks_max]),'Webb':([mu_min,Ks_min],[mu_max,Ks_max]),'Yano':([mu_min,Ks_min],[mu_max,Ks_max]),
        'Haldane':([mu_min,Ks_min],[mu_max,Ks_max])}

        #bounds = ([mu_min,Ks_min],[mu_max,Ks_max])
    else:
        bounds = {'Menten':([1e-18,1e-18],[np.inf,np.inf]),'Inverse_Monod':([1e-18,1e-18],[np.inf,np.inf]),'exp_growth':([1e-18],[np.inf]),
        'Han':([1e-18,1e-18],[np.inf,np.inf]),'Luong':([1e-18,1e-18],[np.inf,np.inf]),'Andrews':([1e-18,1e-18],[np.inf,np.inf]),'Aiba':([1e-18,1e-18],[np.inf,np.inf]),
        'Moser':([1e-18,1e-18],[np.inf,np.inf]),'Edward':([1e-18,1e-18],[np.inf,np.inf]),'Webb':([1e-18,1e-18],[np.inf,np.inf]),'Yano':([1e-18,1e-18],[np.inf,np.inf]),
        'Haldane':([1e-18,1e-18],[np.inf,np.inf])}
        #bounds = ([1e-18,1e-18],[np.inf,np.inf])

    #Create dictionary of additional bounds to be applied to each model, these are inserted into curve fit to set limits to which parameters may fall, with and without
    #estimiated parameters as these additional parameters may not be estimated from menten kenetics theory
    ad_bounds = {'Menten':([],[]),'Inverse_Monod':([],[]),'exp_growth':([],[]),'Han':([Smin,-np.inf,-np.inf],[np.inf,np.inf,np.inf]),'Luong':([Smin,-np.inf],[np.inf,np.inf])
    ,'Andrews':([1e-13],[np.inf]),'Aiba':([0],[np.inf]),'Moser':([0],[np.inf]),'Edward':([1e-13],[np.inf]),'Webb':([1e-13,-np.inf],[np.inf,np.inf]),
    'Yano':([1e-13,-np.inf],[np.inf,np.inf]),'Haldane':([1e-13,Smin],[np.inf,np.inf])}

    return (Smin,bounds,ad_bounds)
