function ipT_resolve_dist10()
  % ARC 变体: 饼间距 dist 3->10mm (vs ARC 3mm), Nc=12/WID=12mm 不变。
  % dist 改两处: (1) Heig=(wid+dist)*Nc=(12+10)*12=264mm(vs 180) -> 更高WP -> 峰值场降 -> Jc升 -> Ip升;
  %             (2) H_magnet=(WID+DIST)*NP 直接变高 -> V_magnet↑ -> 核热↑ (经济侧由 device.py 自动算)。
  % 安匝标定: dist 不入安匝, Ip_cal=8.4e6/(1750*12)=400 (同ARC)。Ic=Jc*wid_cm=Jc*1.2 (WID仍12mm)。
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_dist10.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    Nc=12; AT=8.4e6; lf=0.7; wid_cm=1.2; dist_mm=10;
    Nt_fem = str2double(char(m.param.get('Nt')));
    Ip_cal = AT/(Nt_fem*Nc);
    m.param.set('dist', [num2str(dist_mm) '[mm]']);   % 3 -> 10mm
    m.param.set('Ip', num2str(Ip_cal));
    m.geom('geom1').run(); m.mesh('mesh1').run();
    fprintf('dist=%dmm Nc=%d Nt_fem=%g Ip_cal=%.1f amp_turns=%.3e (Heig=%.0fmm vs ARC 180)\n', ...
            dist_mm,Nc,Nt_fem,Ip_cal,Nt_fem*Nc*Ip_cal,(12+dist_mm)*Nc);
    fprintf(' T[K]  maxB[T]  minJc  Ic(=Jc*1.2)  Ip(0.7)  Ip取整  Nt_design\n');
    M=[];
    for T=[4.2 10 20]
      m.param.set('Top', num2str(T));
      m.sol('sol1').runAll;
      maxB  = mphmax(m,'Bmag_T','volume','selection',2,'dataset','dset1');
      Jcmin = mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
      Ic=Jcmin*wid_cm; Ip=lf*Ic; Ip_r=round(Ip/10)*10; Nt=AT/(Nc*Ip_r);
      fprintf(' %4.1f  %6.1f  %8.0f  %9.0f  %7.0f  %5d  %6.0f\n', T,maxB,Jcmin,Ic,Ip,Ip_r,Nt);
      M=[M; T maxB Jcmin Ic Ip Ip_r Nt];
    end
    writematrix(M,[res 'ip_vs_T_dist10.csv']);
    m.param.set('Top','20'); m.sol('sol1').runAll;
    mphsave(m,[base 'ARC_field_mangiarotti_dist10.mph']);
    fprintf('DIST10_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
