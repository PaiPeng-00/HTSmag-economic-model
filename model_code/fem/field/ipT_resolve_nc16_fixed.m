function ipT_resolve_nc16_fixed()
  % 修正版: 改 Nc 时, 必须把 Ip 标定到使 amp_turns=Nt_fem*Nc*Ip=8.4MA, 否则场随 Nc 错误放大。
  % 场 = Ip*mf.normB, amp_turns=Nt*Nc*Ip. 之前 Nc=16 未改 Ip -> amp_turns~11-13MA -> 场=32T(错)。
  addpath(fullfile(getenv('COMSOL_ROOT'),'mli')); mphstart(2036);
  base=[fullfile(getenv('HTSMAG_FEM_DATA_ROOT'),'1-field-FEM') filesep];
  res=[getenv('HTSMAG_RESULTS_ROOT') filesep];
  diary([res 'ipT_nc16_fixed.log']);
  try
    m=mphload([base 'ARC_field_mangiarotti.mph']);
    Nc=16; AT=8.4e6; lf=0.7;
    Nt_fem = str2double(char(m.param.get('Nt')));   % FEM 径向 Nt (默认2000, 定 dr_TF)
    Ip_cal = AT/(Nt_fem*Nc);                        % 使 amp_turns=8.4MA -> 场标定到 8.4MA
    m.param.set('Nc', num2str(Nc));
    m.param.set('Ip', num2str(Ip_cal));             % 关键修正!
    m.geom('geom1').run(); m.mesh('mesh1').run();
    fprintf('Nc=%d Nt_fem=%g Ip_cal=%.1fA amp_turns=%.3eA (应=8.4e6)\n', Nc, Nt_fem, Ip_cal, Nt_fem*Nc*Ip_cal);
    fprintf(' T[K]  maxB[T](应~23)  minJc  Ic_tape  Ip(0.7)  Ip取整  Nt_design\n');
    M=[];
    for T=[4.2 10 20]
      m.param.set('Top', num2str(T));
      m.sol('sol1').runAll;
      maxB  = mphmax(m,'Bmag_T','volume','selection',2,'dataset','dset1');
      Jcmin = mphmin(m,'Jc_mang','volume','selection',2,'dataset','dset1');
      Ic=1.2*Jcmin; Ip=lf*Ic; Ip_r=round(Ip/10)*10; Nt=AT/(Nc*Ip_r);
      fprintf(' %4.1f  %8.1f  %8.0f  %7.0f  %7.0f  %5d  %6.0f\n', T,maxB,Jcmin,Ic,Ip,Ip_r,Nt);
      M=[M; T maxB Jcmin Ic Ip Ip_r Nt];
    end
    writematrix(M,[res 'ip_vs_T_nc16_fixed.csv']);
    m.param.set('Top','20'); m.sol('sol1').runAll;
    mphsave(m,[base 'ARC_field_mangiarotti_Nc16.mph']);   % 覆盖为修正后(场=8.4MA)
    fprintf('NC16_FIXED_OK\n');
  catch ME
    fprintf(2,'ERR: %s\n', getReport(ME,'basic','hyperlinks','off'));
  end
  diary off;
end
