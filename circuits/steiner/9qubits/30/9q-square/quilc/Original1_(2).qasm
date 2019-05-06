// EXPECTED_REWIRING [0 1 2 3 4 5 6 8 7]
// CURRENT_REWIRING [7 0 2 6 3 4 5 1 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-2.087802470758894*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.3844841619731474*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.2762476260936904*pi) q[5];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rz(-2.9290253115118636*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.0941102315580262*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(3.1399792439357554*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.357753762026864*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.1380715957104617*pi) q[8];
cz q[8], q[3];
rz(-1.569226657032373*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(-1.6597773032328078*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.3769625214783772*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.674236967864048*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.1295875897583096*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.761369489712264*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.9641888827222767*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9438241621069082*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
cz q[4], q[7];
rz(-2.0295582404314327*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.21900946209460065*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.13679811420332033*pi) q[1];
rz(1.6533701433891395*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.1464786462244563*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(-1.5707963267948966*pi) q[1];
rz(0.48390228989904793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-2.3918013978459243*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.49784898005450595*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.8394531973373798*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.427516657759085*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(0.06535081309954816*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
cz q[6], q[7];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[4], q[7];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.467355685725745*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4564375502462912*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4269954866939925*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-1.4888199713396464*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(1.7878236710371715*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.6004291997584115*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.325192285824369*pi) q[2];
cz q[4], q[3];
rz(-0.2135327231440847*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.381184772407101*pi) q[3];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(2.792351161459536*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.3683828410466443*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(3.1366450114882087*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.1390503155539156*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.428564689870547*pi) q[4];
cz q[4], q[1];
rz(1.5637841052497952*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[3];
rz(-0.24863750384238226*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.811408958366952*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5103683338787532*pi) q[4];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.8286668169330742*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.674236967864048*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.053790182830899*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.757108491616645*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-0.7054512992987929*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-2.6377094317664747*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9158762419894946*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.311968121853446*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.6342363084446725*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.7322677345805062*pi) q[7];
cz q[7], q[6];
rz(0.20531889940822623*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.6742369678640496*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.5368438954816568*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.8021778417645038*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.5735168562412467*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.6182766136585056*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.20942646042571056*pi) q[8];
rz(-1.789259171266604*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.5577143832903904*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[4], q[1];
rx(1.5707963267948966*pi) q[0];
cz q[5], q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[0];
rz(-1.467355685725745*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.40743956815618176*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.757108491616646*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.7054512992987938*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(1.1247040673365836*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.4122013514151943*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.7909729511319712*pi) q[0];
rz(-3.139979446518012*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.7812833900598788*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(-1.5707963267948966*pi) q[0];
rz(0.8955662171604528*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(-0.7252104974953506*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.7721287804629995*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.7783992202416927*pi) q[5];
rz(-1.2092137040180684*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.5857701786916701*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[5];
rz(-2.435421591678338*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-0.45807737190892084*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(1.4191219952168934*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.2400307164201363*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.9581089657343345*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6015542728903501*pi) q[4];
rz(-0.4212198777091495*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.2797379313914534*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.6986661788341757*pi) q[5];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.889857456088093*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-1.1645820567151595*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1653856061068779*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.5146654427613733*pi) q[5];
rz(2.974199253551747*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.1220503050325252*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.9541659868615504*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[4];
cz q[5], q[4];
rz(-2.1576005424102283*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.95278435000127*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.9061833662217946*pi) q[6];
cz q[7], q[6];
cz q[3], q[4];
rz(1.0537901828308989*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.3844841619731472*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.27624762609369*pi) q[8];
rz(1.9494059617807098*pi) q[6];
rx(3.141592653589793*pi) q[6];
rz(0.749791255743869*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.5170061439639977*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.3844841619731467*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.436141354291*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
cz q[1], q[0];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[6];
cz q[7], q[4];
rz(-1.8804331874529314*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.549498312287088*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
cz q[1], q[4];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4564375502462912*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4269954866939922*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(3.141592653589793*pi) q[0];
rz(2.217146941469614*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(-1.1645820567151592*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.16538560610687794*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.6269272108284194*pi) q[2];
rz(2.217146941469614*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
rz(1.4739440244072672*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rz(1.261159466136863*pi) q[7];
rx(3.141592653589793*pi) q[7];
rz(-1.4888199713396464*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[8];