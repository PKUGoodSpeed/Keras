#include <cassert>
#include <iomanip>
#include <bits/stdc++.h>

using namespace std;
typedef vector<int> vi;
typedef vector<double> vd;

class LoadRef{
    int N_feat;
    vector<vd> statics;
public:
    LoadRef(const int &n_feat): N_feat(n_feat), statics(vector<vd>(n_feat, vd(2,0.))){}
    void loadData(const vector<string> &fnames){
        vector<vd> feats(N_feat, vd());
        for(auto fname_str: fnames){
            ifstream fin;
            fin.open(fname_str.c_str());
            string info;
            getline(fin, info);
            /* First, load in features */
            while(getline(fin, info)){
                auto j = info.find(' ') + 1;
                double daytime = stod(info.substr(j))*3600. + stod(info.substr(j+3))*60. + stod(info.substr(j+6));
                feats[0].push_back(daytime);
                for(int i=1;i<N_feat;++i){
                    j = info.find(',',j + 1);
                    assert(j != string::npos);
                    feats[i].push_back(stod(info.substr(j+1)));
                }
            }
            fin.close();
        }
        int N_samp = (int)feats[0].size();
        for(int i=0;i<N_feat;++i){
            double ave = accumulate(feats[i].begin(), feats[i].end(), 0.)/N_samp;
            vd tmp(N_samp, 0.);
            transform(feats[i].begin(), feats[i].end(), tmp.begin(), [&ave](double x){return pow(x-ave, 2.);});
            double var = accumulate(tmp.begin(), tmp.end(), 0.)/N_samp;
            statics[i][0] = ave;
            statics[i][1] = sqrt(var);
        }
    }
    vector<vd> getStat(){
        return statics;
    }
};

class GetTrainTest{
    const double inf = 1.E8;
    vd ask, bid;
    int N_samp, N_feat;
    vector<vd> feats;
public:
    GetTrainTest(const int &n_feat): N_feat(n_feat), 
    feats(vector<vd>(n_feat, vd())){}
    void loadData(char *filename, const int N_pass){
        ifstream fin;
        fin.open(filename);
        string info, date = "00000";
        getline(fin, info);
        /* First, load in features */
        while(getline(fin, info)){
            auto j = info.find(' ') + 1;
            double daytime = stod(info.substr(j))*3600. + stod(info.substr(j+3))*60. + stod(info.substr(j+6));
            feats[0].push_back(daytime);
            for(int i=1;i<N_feat;++i){
                j = info.find(',',j + 1);
                assert(j != string::npos);
                feats[i].push_back(stod(info.substr(j+1)));
            }
        }
        fin.close();
        N_samp = (int)feats[0].size();
        cout<<N_samp<<"  shaocong"<<endl;
        ask.resize(N_samp);
        bid.resize(N_samp);
        transform(feats[1].begin(), feats[1].end(), feats[N_feat-1].begin(), ask.begin(), 
        [](double x, double y){return abs(x)+abs(y)/2.;});
        transform(feats[1].begin(), feats[1].end(), feats[N_feat-1].begin(), bid.begin(), 
        [](double x, double y){return abs(x)-abs(y)/2.;});
    }
    int getSellAction(const vd &B, double price_line){
        int ans = 0;
        for(auto k: B) if(k > price_line) ++ans;
        return ans;
    }
    int getBuyAction(const vd &A, double price_line){
        int ans = 0;
        for(auto k: A) if(k < price_line) ++ans;
        return ans;
    }
    void getPerfectAct(vi &per_act, const double fee, const double prof = 64.){
        double low_price = 0., high_price = 500027.5;
        while(low_price < high_price - 0.5){
            double mid_price = (low_price + high_price)/2.;
            int buy_cnt = this->getBuyAction(ask, mid_price);
            int sell_cnt = this->getSellAction(bid, mid_price + prof*fee);
            if(buy_cnt > sell_cnt) high_price = mid_price;
            else low_price = mid_price;
        }
        double mid_price = (low_price + high_price)/2.;
        per_act.resize((int)ask.size());
        for(int i=0;i<(int)ask.size();i++){
            if(ask[i] < mid_price) per_act[i] = 1;
            else if(bid[i] >= mid_price + prof*fee) per_act[i] = 2;
            else per_act[i] = 0;
        }
    }
    
    void normalizeFeats(const vector<vd> &norm){
        for(int i=0;i<N_feat;++i){
            double ave = norm[i][0], std = norm[i][1];
            vd tmp(feats[i]);
            transform(tmp.begin(), tmp.end(), feats[i].begin(), 
            [&ave, &std](double x){return (x-ave)/std;});
        }
    }
    
    vector<vd> getFeats(){
        return feats;
    }
    
    vd getAsk(){
        return ask;
    }
    vd getBid(){
        return bid;
    }
};

int main(int argc, char* argv[]){
    vector<string> fnames = {
        "data/tx_snap_packet_20170501_20170531_fms.csv",
        "data/tx_snap_packet_20170601_20170630_fms.csv"
    };
    LoadRef lref(14);
    lref.loadData(fnames);
    auto ans = lref.getStat();
    for(auto vec: ans) cout<<vec[0]<<' '<<vec[1]<<endl;
    GetTrainTest test(14);
    test.loadData(argv[1], 0);
    vi act;
    test.getPerfectAct(act, 0.5);
    int cnt0 = 0, cnt1 = 0, cnt2 = 0;
    for(auto k: act) {
        if(k==0) ++cnt0;
        if(k==1) ++cnt1;
        if(k==2) ++cnt2;
    }
    cout<<"Number of 0: "<<cnt0<<endl;
    cout<<"Number of 1: "<<cnt1<<endl;
    cout<<"Number of 2: "<<cnt2<<endl;
    test.normalizeFeats(ans);
    auto feats = test.getFeats();
    ofstream fout;
    fout.open("data/train.txt");
    int N_feat = (int)feats.size(), N_samp = (int)feats[0].size();
    for(int i=0;i<N_samp;++i){
        for(int j=0;j<N_feat;++j){
            fout<<feats[j][i]<<' ';
        }
        fout<<act[i];
        if(i<N_samp-1) fout<<endl;
    }
    fout.close();
    cout<<"Finish Generating ./data/train.txt!"<<endl;
    auto ask = test.getAsk(), bid = test.getBid();
    fout.open("data/prcs.txt");
    for(int i=0;i<N_samp;++i){
        fout<<ask[i];
        if(i<N_samp-1) fout<<' ';
        else fout<<endl;
    }
    for(int i=0;i<N_samp;++i){
        fout<<bid[i];
        if(i<N_samp-1) fout<<' ';
    }
    cout<<"Finish Generating ./data/prcs.txt!"<<endl;
    return 0;
}