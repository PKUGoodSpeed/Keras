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
    void loadData(char *filename){
        vector<vd> feats(N_feat, vd());
        ifstream fin;
        fin.open(filename);
        string info;
        getline(fin, info);
        /* First, load in features */
        while(getline(fin, info)){
            auto j = info.find(' ') + 1;
            for(int i=0;i<N_feat;++i){
                j = info.find(',',j + 1);
                assert(j != string::npos);
                feats[i].push_back(stod(info.substr(j+1)));
            }
        }
        int N_samp = (int)feats[0].size();
        fin.close();
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
            auto j = info.find(' ');
            if(info.substr(0,j) != date){
                cout<<date<<endl;
                date = info.substr(0, j);
                for(int i=0;i<N_pass-1;++i) getline(fin, info);
                continue;
            }
            for(int i=0;i<N_feat;++i){
                j = info.find(',',j + 1);
                assert(j != string::npos);
                feats[i].push_back(stod(info.substr(j+1)));
            }
        }
        fin.close();
        N_samp = (int)feats[0].size();
        ask.resize(N_samp);
        bid.resize(N_samp);
        transform(feats[0].begin(), feats[0].end(), feats[N_feat-1].begin(), ask.begin(), 
        [](double x, double y){return abs(x)+abs(y)/2.;});
        transform(feats[0].begin(), feats[0].end(), feats[N_feat-1].begin(), bid.begin(), 
        [](double x, double y){return abs(x)-abs(y)/2.;});
    }
    double getPerfectAct(vi &per_act, const double fee){
        vector<vd> dp(N_samp, vd(3,-inf));
        vector<vi> act(N_samp, vi(3, 0));
        dp[0][1] = 0.;
        dp[0][0] = bid[0] - fee;
        dp[0][2] = -ask[0] - fee;
        for(int i=1;i<N_samp;++i){
            /* for position == 1 */
            dp[i][2] = dp[i-1][2];
            act[i][2] = 2;
            if(dp[i-1][1] - ask[i] - fee >= dp[i][2]){
                dp[i][2] = dp[i-1][1] - ask[i] - fee;
                act[i][2] = 1;
            }
            if(dp[i-1][0] - 2.*ask[i] - fee >= dp[i][2]){
                dp[i][2] = dp[i-1][0] - 2.*ask[i] - fee;
                act[i][2] = 0;
            }
            
            /* for position == 0 */
            dp[i][1] = dp[i-1][1];
            act[i][1] = 1;
            if(dp[i-1][2] + bid[i] - fee >= dp[i][1]){
                dp[i][1] = dp[i-1][2] + bid[i] - fee;
                act[i][1] = 2;
            }
            if(dp[i-1][0] - ask[i] - fee >= dp[i][1]){
                dp[i][1] = dp[i-1][0] - ask[i] - fee;
                act[i][1] = 0;
            }
            
            /* for position == -1 */
            dp[i][0] = dp[i-1][0];
            act[i][0] = 0;
            if(dp[i-1][2] + 2.*bid[i] - fee >= dp[i][0]){
                dp[i][0] = dp[i-1][2] + 2.*bid[i] - fee;
                act[i][0] = 2;
            }
            if(dp[i-1][1] + bid[i] - fee >= dp[i][0]){
                dp[i][0] = dp[i-1][1] + bid[i] - fee;
                act[i][0] = 1;
            }
        }
        vi pos{1};
        for(int i=N_samp-1, tmp_pos = 1; i>0; --i){
            tmp_pos = act[i][tmp_pos];
            pos.push_back(tmp_pos);
        }
        reverse(pos.begin(), pos.end());
        per_act.resize(N_samp);
        cout<<N_samp<<endl;
        per_act[0] = pos[0] - 1;
        for(int i=1;i<N_samp;++i){
            if(pos[i] == pos[i-1]) per_act[i] = 0;
            else if(pos[i] > pos[i-1]) per_act[i] = 1;
            else per_act[i] = -1;
        }
        int i = 0;
        while(i<N_samp){
            if(per_act[i]>0){
                int j = i-1;
                while(j>=0 && ask[j]==ask[i]){
                    assert(!per_act[j]);
                    per_act[j] = per_act[i];
                    --j;
                }
                j = i+1;
                while(j<N_samp && ask[j]==ask[i]){
                    assert(!per_act[j]);
                    per_act[j] = per_act[i];
                    ++j;
                }
                i = j;
            }
            else if(per_act[i]<0){
                int j = i - 1;
                while(j>=0 && bid[j] == bid[i]){
                    assert(!per_act[j]);
                    per_act[j] = per_act[i];
                    --j;
                }
                j = i+1;
                while(j<N_samp && bid[j]==bid[i]){
                    assert(!per_act[j]);
                    per_act[j] = per_act[i];
                    ++j;
                }
                i = j;
            }
            else i+=1;
        }
        return dp[N_samp-1][1];
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
    assert(argc > 2);
    LoadRef lref(13);
    lref.loadData(argv[1]);
    auto ans = lref.getStat();
    for(auto vec: ans) cout<<vec[0]<<' '<<vec[1]<<endl;
    GetTrainTest test(13);
    test.loadData(argv[2], 500);
    vi act;
    double pnl = test.getPerfectAct(act, 0.5);
    cout<<endl<<endl<<"shaocong"<<endl;
    cout<<pnl<<endl;
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
        if(i<N_samp) fout<<' ';
    }
    cout<<"Finish Generating ./data/prcs.txt!"<<endl;
    return 0;
}