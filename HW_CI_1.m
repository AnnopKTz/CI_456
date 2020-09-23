clear; 
data = importdata('Flood_dataset.txt');
data = data.data();
raw_data = Normalize(data);
learning_rate = 0.7;
momentum_rate = 0.1;
min_err = 10.^-5;
err_avg = 1 ;
max_epoch = 1000;
crr_epoch = 1;
hidden_layers = 4;
train_size = size(raw_data,1)-(fix(size(raw_data,1)/10));

w_hi = ones(hidden_layers,size(raw_data,2)-1);
w_oh = 1:hidden_layers;
w_ob = 0;
w_hb = 1:hidden_layers;

validation_set = cross_validation(raw_data);

total_y = zeros(train_size,hidden_layers+1);
total_err = zeros(train_size,max_epoch,10);
rms_err = zeros(10,1);
total_err_test = zeros(size(validation_set,1),1,10);


for i = 1:10
    test_data = validation_set(:,:,i);
    train_data = AddDataTrain(validation_set,i);
    [w_hi,w_oh,w_hb,w_ob] = GenerateW(w_hi,w_oh,w_hb,w_ob);
    %train
    while(crr_epoch <= max_epoch)
        data_rand = randperm(size(train_data,1));
        
        %set all weigth
        if crr_epoch == 1
            [total_w_hi,total_w_oh,total_w_hb,total_w_ob]= SetFirstWeigth(train_size,max_epoch,w_hi,w_oh,w_hb,w_ob);
        end
        
        
        for j = 1 : length(data_rand)
            %-----------------------Feed forward
            dj = train_data(data_rand(j),size(train_data,2));
            vk = 1 : hidden_layers;
            for k = 1 : hidden_layers
                vk(k) = 0;
                for n =1: size(train_data,2)-1
                    vk(k) =  vk(k) + ((train_data(data_rand(j),n)*total_w_hi(j,n,k,crr_epoch)));
                end
                vk(k) = vk(k)+ total_w_hb(j,k,crr_epoch);
                total_y(j,k) = Sigmoid(vk(k));
            end
            yo = 0;
            for m = 1 : hidden_layers
                yo = yo + (total_y(j,m)*total_w_oh(j,m,crr_epoch));
            end
            yo = yo + total_w_ob(j,1,crr_epoch);
            total_y(j,size(total_y,2))=Sigmoid(yo);
            total_err(j,crr_epoch,i)= (dj - total_y(j,size(total_y,2)));
            
            %-----------------------back propagation
            %gradian output
            gd_out = DiffSigmoid(total_y(j,size(total_y,2)))*total_err(j,crr_epoch,i);
            
            %w_oh
            for k = 1 : hidden_layers
                if crr_epoch == 1
                    delta_w_oh = (learning_rate*gd_out*total_y(j,k));
                elseif j==1
                    delta_w_oh = (learning_rate*gd_out*total_y(j,k))+ (momentum_rate*(total_w_oh(size(total_w_oh,1),k,crr_epoch-1)-total_w_oh(j,k,crr_epoch)));
                else
                    delta_w_oh = (learning_rate*gd_out*total_y(j,k))+ (momentum_rate*(total_w_oh(j-1,k,crr_epoch)-total_w_oh(j,k,crr_epoch)));
                end
                
                if j+1 > size(total_w_hb,1)
                    total_w_oh(1,k,crr_epoch+1) = total_w_oh(j,k,crr_epoch) + delta_w_oh;
                else
                    total_w_oh(j+1,k,crr_epoch) = total_w_oh(j,k,crr_epoch) + delta_w_oh;
                end
                
            end
            
            %w_ob
            if crr_epoch == 1
                delta_w_ob = (learning_rate*gd_out*total_w_ob(j,1));
            elseif j ==1
                delta_w_ob = (learning_rate*gd_out*total_w_ob(j,1))+(momentum_rate *(total_w_ob(size(total_w_ob,1),1,crr_epoch-1)-total_w_ob(j,1,crr_epoch)));         
            else
                delta_w_ob = (learning_rate*gd_out*total_w_ob(j,1))+(momentum_rate *(total_w_ob(j-1,1,crr_epoch)-total_w_ob(j,1,crr_epoch)));         
            end
            
            if j+1 > size(total_w_hb,1)
                total_w_ob(1,1,crr_epoch+1) = total_w_ob(j,1,crr_epoch) + delta_w_ob;
            else
                total_w_ob(j+1,1,crr_epoch) = total_w_ob(j,1,crr_epoch) + delta_w_ob;
            end
            
            
            %gradian hidden layers and w_hi
            gd_hid = 1 : hidden_layers;
            for a = 1:length(gd_hid)
                gd_hid(a) = DiffSigmoid(total_y(j,a))*gd_out*total_w_oh(j,a,crr_epoch)*total_y(a);
                for n = 1 : size(train_data,2)-1
                    if crr_epoch == 1
                        delta_w_hi = learning_rate*gd_hid(a)*train_data(n);
                    elseif j ==  1
                        delta_w_hi = (learning_rate*gd_hid(a)*train_data(n))+(momentum_rate * (total_w_hi(size(total_w_hi,1),n,a,crr_epoch-1)-total_w_hi(j,n,a,crr_epoch)));
                    else
                        delta_w_hi = (learning_rate*gd_hid(a)*train_data(n))+(momentum_rate * (total_w_hi(j-1,n,a,crr_epoch)-total_w_hi(j,n,a,crr_epoch)));
                    end
                    
                    if j+1 > size(total_w_hb,1)
                        total_w_hi(1,n,a,crr_epoch+1) = total_w_hi(j,n,a,crr_epoch) + delta_w_hi;
                    else
                        total_w_hi(j+1,n,a,crr_epoch) = total_w_hi(j,n,a,crr_epoch) + delta_w_hi;
                    end
                end
                
                %w_hb
                if crr_epoch == 1
                    delta_w_hb = (learning_rate*gd_out*total_w_hb(j,a));
                elseif j == 1
                    delta_w_hb = (learning_rate*gd_out*total_w_hb(j,a))+(momentum_rate *(total_w_hb(size(total_w_hb,1),a,crr_epoch-1)-total_w_hb(j,a,crr_epoch)));
                else
                    delta_w_hb = (learning_rate*gd_out*total_w_hb(j,a))+(momentum_rate *(total_w_hb(j-1,a,crr_epoch)-total_w_hb(j,a,crr_epoch)));
                end
                
                if j+1 > size(total_w_hb,1)
                    total_w_hb(1,a,crr_epoch+1) = total_w_hb(j,a,crr_epoch) + delta_w_hb;
                else
                    total_w_hb(j+1,a,crr_epoch) = total_w_hb(j,a,crr_epoch) + delta_w_hb;
                end
            end
            
            
        end
        fprintf('THIS EPOCH : %d of validation_set %d\n',crr_epoch,i);
        crr_epoch = crr_epoch+1;
    end
    
    %test
    sum_error = 0;
    for j = 1: size(test_data,1)
        dj = test_data(j,size(test_data,2));
        vk = 1 : hidden_layers;
        yk = 1 : hidden_layers;
        for k = 1 : hidden_layers
            vk(k) = 0;
            yk(k) = 0;
            for n =1: size(test_data,2)-1             
                vk(k) =  vk(k) + ((test_data(j,n))*total_w_hi(1,n,k,crr_epoch));
            end
            vk(k) = vk(k)+ total_w_hb(1,k,crr_epoch);
            yk(k) = Sigmoid(vk(k));
        end
        
        vo = 0;
        for m = 1 : hidden_layers
            vo = vo + (yk(m)*total_w_oh(1,m,crr_epoch));
        end
        vo = vo + total_w_ob(1,1,crr_epoch);
        yo = Sigmoid(vo);
        total_err_test(j,1,i)= abs(dj- yo);
        sum_error = sum_error + (dj- yo).^2;
        
    end
    rms_err(i,1) = sqrt((1/size(test_data,1)*sum_error)) ;
    crr_epoch = 1;
end





%Fuction part
function data_train = cross_validation(x)
    x_rand = randperm(size(x,1));
    size_validation = fix(size(x,1)/10);
    data_train = zeros(size_validation,9,10);
    index = 1;
    in_i = 1;
    in_j = 1;
    for i = 1 : size(x,1)
        for j = 1 : size(x,2)
            data_train(in_i,in_j,index) = x(x_rand(i),j);
            in_j =in_j +1;
        end
        in_i = in_i +1;
        in_j = 1;
        if mod(i,size_validation) == 0
            index = index +1;
            in_i = 1;
        end
    end
end

function data = AddDataTrain(s,n)
    data = zeros((size(s,1)*9+4),size(s,2));
    index = 1;
    for i = 1 : size(s,3)
        if i ~= n && i ~= 11
            for j = 1 : size(s,1)
                for k = 1 : size(s,2)
                    data(index,k) = s(j,k,i);
                end
                index = index + 1;
            end
        elseif i ==11
            for j = 1 : 4
                for k = 1 : size(s,2)
                    data(index,k) = s(j,k,i);
                end
                index = index + 1;
            end
        end
    end
end

function [r_hi,r_oh,r_hb,r_ob]= SetFirstWeigth(n,m,w_hi,w_oh,w_hb,w_ob)
    %create array with i = epoch_round , j = value of each weigth to k = No. of node
    r_hi = zeros(n,size(w_hi,2),size(w_hi,1),m);

    %reate array with i = epoch_round , j = value of weigth to output
    r_oh = zeros(n,length(w_oh),m);

    %create array with i = epoch_round , j = value from bias
    r_hb = zeros(n,length(w_hb),m);
    r_ob = zeros(n,1,m);
    
    for i = 1 : size(w_hi,1)
        for j = 1:size(w_hi,2)
            r_hi(1,j,i,1) = w_hi(i,j);
        end
    end
    
    for i =1: size(w_oh,2)
        r_oh(1,i,1) = w_oh(i);
    end
    
    for i = 1:size(w_hb,2)
        r_hb(1,i,1) = w_hb(i);
    end
    
    r_ob(1,1,1) = w_ob;
        
end

function sigm = Sigmoid(n)
    sigm = 1.0 / (1.0 + exp(-n));
end

function result = DiffSigmoid(n)
    result = n * (1-(n));
end

function data = Normalize(input)
    in_vec = reshape(input,[],1);
    nor_in = normalize(in_vec,'range',[0.1,0.9]);
    data = reshape(nor_in,[size(input,1),size(input,2)]);
end

function [w1,w2,w3,w4] = GenerateW(x,y,z,a)
    w_rand_range = -(1/(sqrt(numel(x)+length(y)+length(z)+1))):0.0001: (1/(sqrt(numel(x)+length(y)+length(z)+1)));
    w_rand_index = randperm(length(w_rand_range));
    
    for i = 1:size(x,1)
        for j = 1:size(x,2)
            x(i,j) = w_rand_range(w_rand_index(1));
            w_rand_index(1)=[];
        end
    end
    
    for i = 1:length(y)
        y(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
    for i = 1:length(z)
        z(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
    a = w_rand_range(w_rand_index(1));
    
    w1 = x;
    w2 = y;
    w3 = z;
    w4 = a;
end
